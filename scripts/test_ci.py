"""CI test script for Ternair advanced features.

Called from .github/workflows/ci.yml after the basic smoke tests pass.
Tests: generation (sampling, streaming, chat), export (safetensors, compression), eval module.
"""

import os
import sys


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
    print("=== CI Advanced Tests ===")
    test_generation()
    test_export()
    test_eval()
    print("=== All CI advanced tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
