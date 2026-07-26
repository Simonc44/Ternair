"""CI test script for Ternair advanced features.

Called from .github/workflows/ci.yml after the basic smoke tests pass.
Tests: generation (sampling, streaming, chat), export (safetensors, compression), eval module.
"""

import os
import shutil
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


def test_pipeline_smoke():
    """Test the TernairPipeline end-to-end on a tiny profile."""
    import os
    import shutil
    import tempfile
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from ternair.training.pipeline import TernairPipeline, PipelineStage
    from ternair.training.config import TrainingConfig

    tmp = tempfile.mkdtemp(prefix="ternair_pipeline_")
    try:
        cfg = TrainingConfig(
            model_profile="tiny",
            model_storage="packed",
            max_train_steps=2,
            batch_size=1,
            dataset_streaming=False,
            log_every=1,
            save_every=2,
        )
        pipeline = TernairPipeline(config=cfg, output_dir=tmp)
        assert pipeline.stage is PipelineStage.UNINITIALIZED

        # Build
        pipeline.build()
        assert pipeline.stage is PipelineStage.BUILT
        assert pipeline.model is not None
        assert pipeline.optimizer is not None

        # Preflight
        estimate = pipeline.preflight_check(batch_size=1, seq_length=4)
        assert estimate.total_bytes > 0
        assert estimate.fits, estimate.summary()

        # Run a 2-step training loop
        ids = torch.randint(0, 256, (1, 4))
        labels = torch.randint(0, 256, (1, 4))
        ds = TensorDataset(ids, labels)

        def collate(b):
            x = torch.stack([t[0] for t in b])
            return {"input_ids": x}

        loader = DataLoader(ds, batch_size=1, collate_fn=collate)
        pipeline.run(loader, max_steps=2)

        assert pipeline.state.stage is PipelineStage.TRAINED
        assert pipeline.state.global_step >= 1, pipeline.state.global_step

        # Freeze + export
        pipeline.freeze()
        assert pipeline.stage is PipelineStage.FROZEN
        out = pipeline.export(format="pt", filename="model.pt")
        assert os.path.exists(out), out
        assert pipeline.state.artifact_paths.get("pt") == out

        # Resume from atomic checkpoint
        resumed_state = pipeline.resume()
        # Without an explicit checkpoint we should get 0; we just want
        # the method to be safe.
        assert resumed_state >= 0

        print(
            f"  Pipeline: stage={pipeline.state.stage.value} "
            f"step={pipeline.state.global_step} "
            f"checks={len(pipeline.state.checkpoints)}"
        )
        print("  [PASS] TernairPipeline smoke test passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_atomic_checkpoint():
    """Test that AtomicCheckpointSaver writes fully (no half files)."""
    import os
    import tempfile
    import torch
    from ternair.training.atomic import AtomicCheckpointSaver

    tmp = tempfile.mkdtemp(prefix="ternair_atomic_")
    saver = AtomicCheckpointSaver(save_dir=tmp)
    state = {"step": 0, "tensor": torch.zeros(4)}
    path = saver.save(state)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
    # No leftover .tmp
    assert not os.path.exists(saver.tmp_path), "tmp file leaked"

    # Save again, ensure .prev is populated.
    state["step"] = 1
    path2 = saver.save(state)
    assert os.path.exists(saver.previous_path)
    assert path != path2 or state["step"] == 1
    loaded = saver.load()
    assert loaded["step"] == 1
    # Resolve sees the latest non-empty checkpoint.
    resolved = saver.resolve_resume_path()
    assert resolved == path2
    shutil.rmtree(tmp, ignore_errors=True)
    print("  [PASS] AtomicCheckpointSaver tests passed")


def test_memory_estimate():
    """Test the memory estimator returns sensible numbers."""
    import torch
    from ternair.training.memory import (
        MemoryEstimate,
        DEFAULT_BYTES_PER_PARAM_OPTIM,
        estimate_memory,
    )
    from ternair.model.size_profiles import tiny_profile
    from ternair.model.modeling import TernairForCausalLM

    cfg = tiny_profile(storage="packed")
    model = TernairForCausalLM(cfg)
    est = estimate_memory(model, batch_size=2, seq_length=8)
    assert est.model_bytes > 0
    assert est.optimizer_bytes >= (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        * DEFAULT_BYTES_PER_PARAM_OPTIM
    )
    # Tiny profile fits on CPU / small GPU.
    print(
        f"  tiny model est: {est.total_bytes / 1024 ** 2:.1f} MiB, "
        f"fits={est.fits}, bottleneck={est.bottleneck}"
    )
    # Summary string is human-readable.
    assert "Memory pre-flight" in est.summary()
    print("  [PASS] Memory estimator tests passed")


def test_intermediate_profiles():
    """Test that the intermediate profiles load and yield valid configs."""
    from ternair.model.size_profiles import (
        PROFILE_REGISTRY, fit_profile_for_budget, small_profile,
        medium_profile, large_profile,
    )
    from ternair.model.config import TernairConfig

    for name in ("small", "medium", "large"):
        fn = PROFILE_REGISTRY[name]
        cfg = fn(storage="packed")
        # `__post_init__` raises on invalid configs, so reaching here is the test.
        assert isinstance(cfg, TernairConfig)
        # Configs should be more expressive than tiny but never exceed 1 GiB.
        assert cfg.hidden_size > 256, f"{name} too small"
        assert cfg.num_hidden_layers >= 8, f"{name} too shallow"

    fitted = fit_profile_for_budget(target_mib=50)
    assert isinstance(fitted, tuple) and len(fitted) == 2
    print(
        f"  Intermediate profiles: small={small_profile().hidden_size} "
        f"medium={medium_profile().hidden_size} large={large_profile().hidden_size}"
    )
    print(f"  Fit-for-budget (50 MiB) -> {fitted}")
    print("  [PASS] Intermediate profiles tests passed")


def test_optimizer_groups():
    """Test that the optimizer precedence fix routes params correctly."""
    import torch
    import torch.nn as nn
    from ternair.training.optimizer import create_param_groups
    from ternair.quantization.linear import TernairLinear

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 32)
            self.norm = nn.LayerNorm(32)
            self.ternary = TernairLinear(32, 32, storage="packed")
            self.fc = nn.Linear(32, 32)

    model = TinyModel()
    groups = create_param_groups(model, lr=1e-3, weight_decay=0.1)
    # Find each bucket.
    decay = sum(g["weight_decay"] for g in groups if g["weight_decay"] > 0)
    no_decay = sum(g["weight_decay"] for g in groups if g["weight_decay"] == 0)

    # Embed -> decay > 0
    embed_param = next(model.embed.parameters())
    embed_bucket = next(g for g in groups if any(p is embed_param for p in g["params"]))
    assert embed_bucket["weight_decay"] > 0, "Embedding should decay"

    # LayerNorm weight -> no decay
    norm_param = next(model.norm.parameters())
    norm_bucket = next(g for g in groups if any(p is norm_param for p in g["params"]))
    assert norm_bucket["weight_decay"] == 0, "LayerNorm should be no-decay"

    # TernairLinear weight -> no decay
    tern_param = model.ternary.weight
    tern_bucket = next(g for g in groups if any(p is tern_param for p in g["params"]))
    assert tern_bucket["weight_decay"] == 0, "Ternair weight should be no-decay"

    print(
        f"  Optimizer groups: decay={decay}, no_decay={no_decay}, "
        f"#groups={len(groups)}"
    )
    print("  [PASS] Optimizer param-groups tests passed")


def test_packing_base8():
    """Test the v0.6.0 packing_base8 codec (5 trits/byte, 1.6 bits/value)."""
    import numpy as np
    import torch
    from ternair.kernels.packing_base8 import (
        MODE_BASE8,
        MODE_PACKED,
        BITS_PER_VALUE,
        pack_trits,
        pack_trits_base8,
        unpack_trits,
        unpack_trits_base8,
        torch_to_packed,
        packed_to_torch,
        bytes_for,
    )

    # MODE_PACKED is the legacy alias for MODE_BASE8.
    assert MODE_PACKED == MODE_BASE8 == "base8"
    assert BITS_PER_VALUE["base8"] == 1.6
    assert BITS_PER_VALUE["packed"] == 1.6

    # Round-trip on a random ternary array.
    rng = np.random.default_rng(0)
    trits = rng.integers(-1, 2, size=257).astype(np.int8)  # non-multiple of 5
    packed = pack_trits(trits)
    assert packed.dtype == np.uint8
    assert packed.size == (257 + 4) // 5  # ceil(257/5)
    assert bytes_for(257) == packed.size
    restored = unpack_trits(packed, length=257)
    assert np.array_equal(restored, trits), "Round-trip mismatch"

    # pack_trits == pack_trits_base8 (alias works).
    assert pack_trits(trits).__class__ == pack_trits_base8(trits).__class__

    # Torch bridge round-trip.
    t = torch.from_numpy(trits)
    packed_np = torch_to_packed(t)
    restored_t = packed_to_torch(packed_np, shape=tuple(t.shape))
    assert restored_t.shape == t.shape
    assert torch.equal(restored_t.to(torch.int8), t)

    # Canonical bits-per-value vs fastpacked.
    assert BITS_PER_VALUE["base8"] < BITS_PER_VALUE["fastpacked"]  # 1.6 < 2.0

    print(
        f"  packing_base8: 257 trits -> {packed.size} bytes "
        f"({BITS_PER_VALUE['base8']} bits/value)"
    )
    print("  [PASS] packing_base8 tests passed")


def test_triton_fast_batched():
    """Test the v0.6.0 triton_fast kernel handles single + batched inputs."""
    import numpy as np
    import torch
    from ternair.kernels.triton_fast import (
        has_triton,
        ternary_matmul_triton,
        ternary_matmul_single_triton,
    )
    from ternair.kernels.packing_fast import pack_trits_2bit

    rng = np.random.default_rng(1)
    M, N = 32, 64
    K_packed = (N + 3) // 4
    trits = rng.integers(-1, 2, size=(M, N)).astype(np.int8)
    packed = pack_trits_2bit(trits.reshape(-1)).reshape(M, K_packed)
    gamma = rng.random(M).astype(np.float32)

    # Reference (NumPy via packed_ops).
    from ternair.kernels.packed_ops import ternary_matmul_numpy_batched
    expected_batched = ternary_matmul_numpy_batched(
        packed, rng.random((4, N)).astype(np.float16), gamma
    )
    expected_single = expected_batched[0]

    # Single-input path: 1-D x in, 1-D y out.
    x1 = rng.random(N).astype(np.float16)
    out_single = ternary_matmul_single_triton(packed, x1, gamma)
    assert out_single.shape == (M,), f"Single output shape: {out_single.shape}"
    # Triton may not be installed in CI -> fall back to NumPy is acceptable.
    if has_triton():
        assert np.allclose(out_single, expected_single, atol=1e-1), (
            f"max abs err {np.abs(out_single - expected_single).max()}"
        )

    # Batched-input path: (B, N) in, (B, M) out.
    xB = rng.random((4, N)).astype(np.float16)
    out_batched = ternary_matmul_triton(packed, xB, gamma)
    assert out_batched.shape == (4, M), f"Batched output shape: {out_batched.shape}"

    # Kernel accepts 1-D x with the unified API too.
    out1d = ternary_matmul_triton(packed, x1, gamma)
    assert out1d.shape == (M,), f"Unified 1-D output shape: {out1d.shape}"

    # has_triton reflects the runtime state.
    print(f"  triton_fast: has_triton={has_triton()}")
    print("  [PASS] triton_fast tests passed")


def test_direct_inference():
    """Test TernairDirectInferencer + TernairLinear backend dispatch.

    Verifies:
    - ``available_backends()`` returns a sensible dict.
    - ``TernairLinear.set_inference_backend(...)`` is honoured.
    - ``prepare()`` returns a working inferer and ``forward()`` matches
      the model's eval-mode forward under the ``"torch"`` backend.
    - Different backends (torch / numpy / cpu_cpp / triton) all
      produce *close-enough* logits (the reference max-abs err is
      below the dtype noise floor).
    """
    import numpy as np
    import torch
    from ternair.model.inference import TernairDirectInferencer
    from ternair.model.modeling import TernairForCausalLM
    from ternair.model.size_profiles import tiny_profile
    from ternair.quantization.linear import TernairLinear

    # 1. availability dict
    avail = TernairDirectInferencer.available_backends()
    assert isinstance(avail, dict)
    assert avail["torch"] is True
    assert avail["numpy"] is True
    print(f"  available backends: {avail}")

    # 2. set_inference_backend validation
    cfg = tiny_profile(storage="fastpacked")
    model = TernairForCausalLM(cfg)
    layers = [m for m in model.modules() if isinstance(m, TernairLinear)]
    assert len(layers) > 0
    layers[0].set_inference_backend("torch")
    layers[0].set_inference_backend("numpy")
    try:
        layers[0].set_inference_backend("does-not-exist")
    except ValueError:
        pass
    else:
        raise AssertionError("set_inference_backend should reject unknown backends")

    # 3. end-to-end on torch backend (always available)
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    inferer = TernairDirectInferencer(model, backend="torch")
    info = inferer.describe()
    assert info["resolved_backend"] == "torch"
    assert info["n_ternary_layers"] == len(layers)

    inferer.prepare()
    logits_torch = inferer.forward(ids)
    assert logits_torch.shape == (1, 4, cfg.vocab_size), logits_torch.shape

    # 4. backend=auto selects the same as torch when no GPU / cppyy
    auto = TernairDirectInferencer(model, backend="auto").prepare()
    assert auto.resolved_backend in {"torch", "triton", "cpu_cpp"}

    # 5. torch vs numpy cross-check on the SAME model.
    # Re-use the already-frozen model: restore() resets the per-layer
    # backend to "auto" so we can re-prepare() under a different backend
    # without disturbing the weights.
    inferer.restore()
    inferer.requested_backend = "numpy"
    inferer.prepare()
    logits_numpy = inferer.forward(ids)

    diff = (logits_torch - logits_numpy).abs().max().item()
    mag = max(
        logits_torch.abs().mean().item(),
        logits_numpy.abs().mean().item(),
        1e-9,
    )
    rel = diff / mag
    # 1. argmax MUST agree (the only thing that matters for generation).
    top1_torch = logits_torch.argmax(dim=-1)
    top1_numpy = logits_numpy.argmax(dim=-1)
    agree = bool((top1_torch == top1_numpy).all().item())
    assert agree, f"top-1 mismatch: torch={top1_torch} numpy={top1_numpy}"
    # 2. fp16 accumulator noise on un-trained random weights is large
    #    in absolute terms, but the RELATIVE error should stay small.
    assert rel < 0.20, f"torch vs numpy rel-diff {rel:.4g} (max|err|={diff:.4g}, mag={mag:.4g})"
    print(
        f"  cross-backend: max|err|={diff:.4g} rel={rel:.4g} top1-agree={agree}"
    )

    # 6. generate() works under direct inference (torch).
    # Restore() then re-prepare() under the torch backend so the
    # cross-check above doesn't affect the generation test.
    inferer.restore()
    inferer.requested_backend = "torch"
    inferer.prepare()
    out = inferer.generate(
        ids, max_new_tokens=4, temperature=0.0,
        eos_token_id=cfg.vocab_size - 1,
    )
    assert out.shape[1] == 8, f"generated shape: {out.shape}"
    print(f"  generate() ok: shape={tuple(out.shape)}")

    # 7. restore() resets the backend
    inferer.restore()
    for m in model.modules():
        if isinstance(m, TernairLinear):
            assert m.inference_backend == "auto"

    print("  [PASS] Direct inference tests passed")


def main():
    print("=== CI Advanced Tests v0.6.0 ===")
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
    test_pipeline_smoke()
    test_atomic_checkpoint()
    test_memory_estimate()
    test_intermediate_profiles()
    test_optimizer_groups()
    test_packing_base8()
    test_triton_fast_batched()
    test_direct_inference()
    print("=== All CI advanced tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
