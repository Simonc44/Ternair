"""CI test script for Ternair advanced features.

Called from .github/workflows/ci.yml after the basic smoke tests pass.
Tests: generation (sampling, streaming, chat), export (safetensors, compression), eval module.
"""

import os
import sys


def test_bitdelta_lora():
    """Test T-LoRA / BitDelta adapters."""
    import torch
    from ternair.quantization.bitdelta import (
        TernaryLoRALinear, TernaryDeltaLinear,
        AdapterRegistry, add_lora_to_model, add_bitdelta_to_model
    )

    # Test TernaryLoRALinear forward
    lora = TernaryLoRALinear(64, 128, rank=8)
    x = torch.randn(2, 16, 64)
    out = lora(x)
    assert out.shape == (2, 16, 128), f"LoRA shape: {out.shape}"
    print(f"  LoRA: {sum(p.numel() for p in lora.parameters())} params")

    # Test TernaryDeltaLinear forward
    delta = TernaryDeltaLinear(64, 128)
    out = delta(x)
    assert out.shape == (2, 16, 128), f"Delta shape: {out.shape}"

    # Test Registry
    import torch.nn as nn
    model = nn.Linear(64, 128)
    registry = AdapterRegistry()
    lora2 = TernaryLoRALinear(64, 128, rank=4)
    registry.register("test", lora2)
    registry.attach(model)
    out = model(x)
    assert out.shape == (2, 16, 128), f"Registry shape: {out.shape}"
    registry.detach()

    print("  [PASS] T-LoRA / BitDelta tests passed")


def test_omni_quant():
    """Test ScaleEquivalence (OmniQuant)."""
    import torch
    from ternair.quantization.activation import ScaleEquivalence

    scale = ScaleEquivalence(256)
    x = torch.randn(2, 16, 256)
    w = torch.randn(128, 256)
    x_s, w_s = scale(x, w)
    assert x_s.shape == x.shape, f"x_s shape: {x_s.shape}"
    assert w_s.shape == w.shape, f"w_s shape: {w_s.shape}"
    print(f"  ScaleEquivalence: scale.mean={scale.scale.mean().item():.4f}")
    print("  [PASS] OmniQuant tests passed")


def test_kv_cache():
    """Test KV-Cache quantization (BitAttention)."""
    import torch
    from ternair.model.attention import _quantize_kv

    k = torch.randn(1, 4, 64, 128)
    k_q = _quantize_kv(k, bits=2)
    assert k_q.shape == k.shape, f"KV quant shape: {k_q.shape}"
    error = (k - k_q).abs().mean().item()
    print(f"  KV quant 2-bit error: {error:.4f}")

    from ternair.model.config import TernairConfig
    cfg = TernairConfig(hidden_size=256, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4,
                         kv_cache_bits=2)
    assert cfg.kv_cache_bits == 2
    print("  [PASS] BitAttention tests passed")


def test_moe():
    """Test Ternary MoE."""
    import torch
    from ternair.model.moe import TernaryMoEBlock
    from ternair.model.config import TernairConfig

    cfg = TernairConfig(hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=4, storage="fastpacked")
    moe = TernaryMoEBlock(cfg, num_experts=4, top_k=2,
                          hidden_size=128, intermediate_size=256)
    x = torch.randn(2, 8, 128)
    out = moe(x)
    assert out.shape == x.shape, f"MoE shape: {out.shape}"
    print(f"  MoE: {sum(p.numel() for p in moe.parameters()):,} params")
    print("  [PASS] Ternary MoE tests passed")


def test_webgpu():
    """Test WebGPU backend kernel generation."""
    from ternair.kernels.webternair import (
        generate_wgsl_ternary_matmul,
        generate_wgsl_rms_norm,
        validate_webgpu_kernels,
    )

    wgsl = generate_wgsl_ternary_matmul()
    assert "@compute" in wgsl
    assert "decode_byte" in wgsl

    rms = generate_wgsl_rms_norm()
    assert "RMSNorm" in rms or "rms" in rms.lower()

    results = validate_webgpu_kernels()
    assert all(results.values()), f"WGSL validation: {results}"
    print(f"  WebGPU shaders valides: {sum(1 for v in results.values() if v)}/{len(results)}")
    print("  [PASS] WebGPU backend tests passed")


def test_gguf_export():
    """Test GGUF export module."""
    import os
    import numpy as np
    from ternair.kernels.gguf_export import export_to_gguf

    # Small test: export a dict of tensors
    tensors = {
        "test.weight": np.random.randn(64, 64).astype(np.float32),
        "test.bias": np.random.randn(64).astype(np.float32),
    }
    config = {"hidden_size": 64, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2,
              "intermediate_size": 128, "vocab_size": 1000,
              "max_position_embeddings": 512}

    path = export_to_gguf(tensors, "/tmp/test.gguf", config=config)
    assert os.path.exists(path), "GGUF file not created"
    size = os.path.getsize(path)
    assert size > 0, f"GGUF file empty: {size}"
    print(f"  GGUF export: {size / 1024:.1f} KiB")
    print("  [PASS] GGUF export tests passed")


def test_triton_fused():
    """Test enhanced Triton fused kernels (module load)."""
    from ternair.kernels.triton_matmul_fused import (
        has_triton, benchmark_triton_vs_numpy
    )
    if has_triton():
        print("  Triton disponible")
        # Only run benchmark on GPU
        import torch
        if torch.cuda.is_available():
            results = benchmark_triton_vs_numpy(M=512, N=512, batch=1, num_warmup=1, num_runs=2)
            print(f"  Speedup: {results.get('speedup', 'N/A')}x")
    else:
        print("  Triton non disponible (skip)")
    print("  [PASS] Triton fused module tests passed")


def test_generation():
    """Test advanced generation: greedy, sampling, streaming, chat templates."""
    import torch
    from ternair.model.size_profiles import tiny_profile
    from ternair.model.modeling import TernairForCausalLM
    from ternair.model.generation import generate, generate_stream, format_chat_prompt

    cfg = tiny_profile(storage="fastpacked")
    model = TernairForCausalLM(cfg)
    ids = torch.tensor([[1, 2, 3, 4, 5]])

    # Greedy
    out = generate(model, ids, max_new_tokens=4, temperature=0.0)
    assert out.shape[1] == 9, f"Greedy shape: {out.shape}"

    # Sampling with repetition penalty
    out = generate(
        model, ids, max_new_tokens=4, temperature=0.8,
        top_k=40, top_p=0.9, repetition_penalty=1.1,
    )
    assert out.shape[1] >= 5, f"Sampling shape: {out.shape}"

    # Streaming
    tokens = [t.item() for t in generate_stream(model, ids, max_new_tokens=3)]
    assert len(tokens) == 3, f"Stream tokens: {len(tokens)}"

    # Chat templates
    chatml = format_chat_prompt(
        [{"role": "user", "content": "Hi"}], format="chatml"
    )
    assert "<|im_start|>" in chatml, f"ChatML: {chatml}"

    llama3 = format_chat_prompt(
        [{"role": "user", "content": "Hi"}], format="llama3"
    )
    assert "<|start_header_id|>" in llama3, f"Llama3: {llama3}"

    print("  [PASS] Advanced generation tests passed")


def test_export():
    """Test safetensors export and compression report."""
    import torch
    import os
    from ternair.model.size_profiles import tiny_profile
    from ternair.model.modeling import TernairForCausalLM
    from ternair.model.export import (
        export_to_safetensors,
        compute_compression_report,
        build_hf_config,
    )

    cfg = tiny_profile(storage="fastpacked")
    model = TernairForCausalLM(cfg)
    model.freeze_storage()
    model.eval()

    # Compression report
    report = compute_compression_report(model)
    assert report["compression_ratio"] > 1.0, (
        f"Compression ratio: {report['compression_ratio']}"
    )
    assert report["savings_percent"] > 50.0, (
        f"Savings: {report['savings_percent']}%"
    )
    print(
        f"  Compression: {report['compression_ratio']}x, "
        f"savings {report['savings_percent']}%"
    )

    # Export safetensors
    path = export_to_safetensors(model, "/tmp/test_model.safetensors")
    assert os.path.exists(path), "File not created"
    file_size = os.path.getsize(path)
    print(f"  Safetensors: {file_size / 1024:.1f} KiB")

    # HF config
    hf_cfg = build_hf_config(cfg)
    assert hf_cfg["model_type"] == "ternair"
    qc = hf_cfg["quantization_config"]
    assert qc["quantization_method"] == "bitnet_b1_58"

    print("  [PASS] Export tests passed")


def test_eval():
    """Test evaluation module structure and report serialization."""
    from ternair.benchmark.eval import (
        PerplexityResult,
        ZeroShotResult,
        SpeedResult,
        EvalReport,
    )

    report = EvalReport()
    report.perplexity.append(PerplexityResult(
        dataset="wikitext-2/test", num_tokens=5000,
        perplexity=12.5, cross_entropy=2.53,
        num_batches=10, time_seconds=0.5,
    ))
    report.zero_shot.append(ZeroShotResult(
        benchmark="HellaSwag", num_samples=200,
        accuracy=0.425, num_correct=85,
    ))
    report.speed = SpeedResult(
        tokens_per_second=15.2, memory_mib=450.0,
        prompt_length=10, generated_length=32,
        wall_time_seconds=2.1,
    )

    data = report.to_dict()
    assert "perplexity" in data
    assert "zero_shot" in data
    assert "speed" in data
    assert data["perplexity"][0]["perplexity"] == 12.5
    assert data["zero_shot"][0]["accuracy"] == 0.425
    assert data["speed"]["tokens_per_second"] == 15.2

    print("  [PASS] Eval module tests passed")


def main():
    print("=== CI Advanced Tests v0.4.0 ===")
    test_generation()
    test_export()
    test_eval()
    test_bitdelta_lora()
    test_omni_quant()
    test_kv_cache()
    test_moe()
    test_webgpu()
    test_gguf_export()
    test_triton_fused()
    print("=== All CI advanced tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
