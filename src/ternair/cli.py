"""Command-line entry point - ``python -m ternair``."""

from __future__ import annotations

import argparse
import sys

from ternair.benchmark.size import describe, fit_one_gb
from ternair.model.size_profiles import base_profile, one_gb_profile, tiny_profile
from ternair.quantization.packing import BITS_PER_VALUE


PROFILES = {
    "tiny": tiny_profile,
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ternair", description="BitNet-b1.58-style ternary LM")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--storage", choices=["packed", "int8"], default="packed")
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
