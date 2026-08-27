"""Command-line entry point - ``python -m ternair``."""

from __future__ import annotations

import argparse
import sys

from ternair.benchmark.size import describe, fit_one_gb
from ternair.model.size_profiles import base_profile, one_gb_profile, tiny_profile
from ternair.kernels.packing_base8 import BITS_PER_VALUE


from ternair.model.size_profiles import small_profile, medium_profile, large_profile

PROFILES = {
    "tiny": tiny_profile,
    "small": small_profile,
    "medium": medium_profile,
    "large": large_profile,
    "base": base_profile,
    "one_gb": one_gb_profile,
}


def _info(args: argparse.Namespace) -> int:
    cfg = PROFILES[args.profile](storage=args.storage)
    if args.fit_one_gb:
        cfg = fit_one_gb(cfg)
    print(cfg.to_dict())
    return 0


def _size(args: argparse.Namespace) -> int:
    cfg = PROFILES[args.profile](storage=args.storage)
    if args.fit_one_gb:
        cfg = fit_one_gb(cfg)
    print(describe(cfg, embedding_dtype_bytes=args.embedding_dtype_bytes))
    return 0


def _demo(args: argparse.Namespace) -> int:
    import torch

    from ternair.model.modeling import TernairForCausalLM
    from ternair.model.generation import generate
    from ternair.training.data import CharTokenizer, DEFAULT_CORPUS

    profile = PROFILES[args.profile](storage=args.storage)
    if profile.num_hidden_layers > 12 and not args.allow_big:
        print(
            f"[warn] profile {args.profile!r} has "
            f"{profile.num_hidden_layers} layers; this demo will be slow on CPU. "
            "Re-run with --allow-big if you want to proceed.",
            file=sys.stderr,
        )
        return 2

    print(f"Building ternair model ({args.profile}, storage={args.storage})…")
    model = TernairForCausalLM(profile)
    print(f"  ↳ ternary params : {model.count_parameters():,}")

    # Smoke forward
    tok = CharTokenizer(DEFAULT_CORPUS)
    ids = torch.tensor([tok.bos_id] + tok.encode("hello world"), dtype=torch.long).unsqueeze(0)
    out = model(ids)
    print(f"  ↳ forward logits shape: {tuple(out.shape)}")

    # Freeze and re-forward to prove the inference path works.
    print("Calling freeze_storage() to switch to packed trit buffer …")
    model.freeze_storage()
    model.eval()
    out2 = model(ids)
    print(f"  ↳ post-freeze logits shape: {tuple(out2.shape)}")

    if args.max_new_tokens > 0:
        generated = generate(model, ids, max_new_tokens=args.max_new_tokens, eos_token_id=tok.eos_id)
        print("Generated tokens :", generated[0].tolist())
        print("Decoded text     :", repr(tok.decode(generated[0].tolist())))

    print(f"Projected size : {model.num_bytes() / 1024**2:.1f} MiB")
    return 0


def _train_one(args: argparse.Namespace) -> int:
    import torch

    from ternair.model.modeling import TernairForCausalLM
    from ternair.training.trainer import train_one_step
    from ternair.training.data import tokenise_corpus

    cfg = PROFILES[args.profile](storage=args.storage)
    model = TernairForCausalLM(cfg)
    ids, tok = tokenise_corpus()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss, _ = train_one_step(model, ids, optimizer=optimizer)
    print(f"profile={args.profile}  loss={loss:.4f}  vocab={tok.vocab_size}")
    print(f"params including embedding: {model.count_parameters(include_embedding=True):,}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    """Start the HTTP inference server."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from ternair.server import serve
    serve(
        profile_name=args.profile,
        host=args.host,
        port=args.port,
        storage=args.storage,
    )
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    """Run reproducible benchmarks."""
    from ternair.benchmark.reproducible import run_benchmark
    result = run_benchmark(
        profile=args.profile,
        storage=args.storage,
        device=args.device,
        run_perplexity=not args.skip_perplexity,
        run_speed=not args.skip_speed,
        eval_tokens=args.eval_tokens,
    )
    print(result.summary())
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Results saved to {args.output}")
    return 0


def _infer(args: argparse.Namespace) -> int:
    """Direct-inference CLI -- uses TernairDirectInferencer + the kernels."""
    import torch

    from ternair.model.modeling import TernairForCausalLM
    from ternair.model.inference import TernairDirectInferencer
    from ternair.training.data import CharTokenizer, DEFAULT_CORPUS

    profile = PROFILES[args.profile](storage=args.storage)
    if profile.num_hidden_layers > 12 and not args.allow_big:
        print(
            f"[warn] profile {args.profile!r} has "
            f"{profile.num_hidden_layers} layers; this demo will be slow on CPU. "
            "Re-run with --allow-big if you want to proceed.",
            file=sys.stderr,
        )
        return 2

    print(f"Building ternair model ({args.profile}, storage={args.storage})…")
    model = TernairForCausalLM(profile)
    print(f"  ↳ ternary params : {model.count_parameters():,}")

    # Wire the direct-inference backend (auto resolves to triton/cpu_cpp/torch).
    inferer = TernairDirectInferencer(model, backend=args.backend, device=args.device)
    info = inferer.describe()
    print(
        f"  ↳ backend: requested={info['requested_backend']} "
        f"resolved={info['resolved_backend']} "
        f"device={info['device']} "
        f"ternary_layers={info['n_ternary_layers']}"
    )

    tok = CharTokenizer(DEFAULT_CORPUS)
    ids = torch.tensor([tok.bos_id] + tok.encode(args.prompt), dtype=torch.long).unsqueeze(0)

    # Smoke forward
    inferer.prepare()
    logits = inferer.forward(ids)
    print(f"  ↳ forward logits shape: {tuple(logits.shape)}")

    if args.max_new_tokens > 0:
        out = inferer.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=tok.eos_id,
        )
        text = tok.decode(out[0].tolist())
        print(f"  ↳ generated tokens : {out[0].tolist()}")
        print(f"  ↳ decoded text     : {text!r}")

    print(
        f"  ↳ backend stats : available={TernairDirectInferencer.available_backends()}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ternair", description="BitNet-b1.58-style ternary LM")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--storage",
        choices=["packed", "int8", "fastpacked"],
        default="packed",
        help="Packed storage format. fastpacked (4 trits/byte) is required for kernel backends.",
    )
    common.add_argument(
        "--profile",
        choices=list(PROFILES),
        default="tiny",
        help="Which preset to use (tiny / base / one_gb).",
    )
    common.add_argument(
        "--fit-one-gb",
        action="store_true",
        help="Auto-tune num_hidden_layers so the projection lands under 1 GiB.",
    )

    p_info = sub.add_parser("info", parents=[common], help="Print a config in JSON-ish form.")
    p_info.set_defaults(func=_info)

    p_size = sub.add_parser("size", parents=[common], help="Show the projected size breakdown.")
    p_size.add_argument("--embedding-dtype-bytes", type=int, default=2, help="1 or 2 (FP16 by default)")
    p_size.set_defaults(func=_size)

    p_demo = sub.add_parser("demo", parents=[common], help="Build + forward a tiny model end-to-end.")
    p_demo.add_argument("--max-new-tokens", type=int, default=16)
    p_demo.add_argument("--allow-big", action="store_true", help="Allow building larger profiles.")
    p_demo.set_defaults(func=_demo)

    p_train = sub.add_parser(
        "train-one",
        parents=[common],
        help="Run ONE Adam step on the toy corpus (smoke test).",
    )
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.set_defaults(func=_train_one)

    p_infer = sub.add_parser(
        "infer",
        parents=[common],
        help=(
            "Direct inference via TernairDirectInferencer "
            "(auto-selects triton / cpu_cpp / torch kernels)."
        ),
    )
    # Override the storage default for `infer`: the kernel backends only
    # handle ``fastpacked`` (4 trits/byte). Force it here so users don't
    # have to know the implementation detail.
    p_infer.set_defaults(storage="fastpacked")
    p_infer.add_argument(
        "--backend",
        choices=["auto", "torch", "triton", "cpu_cpp", "numpy"],
        default="auto",
        help="Which kernel to dispatch the ternary matmul to.",
    )
    p_infer.add_argument(
        "--device",
        default=None,
        help="Optional device override (cuda / cpu). Default = model device.",
    )
    p_infer.add_argument("--max-new-tokens", type=int, default=16)
    p_infer.add_argument("--temperature", type=float, default=1.0)
    p_infer.add_argument("--top-k", type=int, default=0)
    p_infer.add_argument("--top-p", type=float, default=0.0)
    p_infer.add_argument("--prompt", default="hello world")
    p_infer.add_argument(
        "--allow-big", action="store_true",
        help="Allow building larger profiles (slow on CPU).",
    )
    p_infer.set_defaults(func=_infer)

    p_serve = sub.add_parser(
        "serve",
        parents=[common],
        help="Start OpenAI-compatible HTTP inference server.",
    )
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind address.")
    p_serve.add_argument("--port", type=int, default=8080, help="Bind port.")
    p_serve.set_defaults(func=_serve)

    p_bench = sub.add_parser(
        "benchmark",
        parents=[common],
        help="Run reproducible benchmarks (perplexity + speed).",
    )
    p_bench.add_argument("--device", default="cpu", help="cpu or cuda.")
    p_bench.add_argument("--eval-tokens", type=int, default=1024, help="Tokens for perplexity.")
    p_bench.add_argument("--skip-perplexity", action="store_true")
    p_bench.add_argument("--skip-speed", action="store_true")
    p_bench.add_argument("--output", default=None, help="Save JSON results.")
    p_bench.set_defaults(func=_benchmark)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cmd = args.command
    if cmd is None:
        build_parser().print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
