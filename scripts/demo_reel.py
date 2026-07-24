#!/usr/bin/env python3
"""
Demo reel : entrainement d'un modele Ternair sur des donnees reelles
puis generation de texte apres gel en stockage ternaire.

Ce script montre le workflow complet :
  1. Chargement de donnees texte (corpus integre + option dataset HuggingFace)
  2. Construction d'un modele ternaire (profil tiny = 2,6 M parametres)
  3. Entrainement sur N etapes (configurable)
  4. Gel en stockage ternaire compacte (packed 1,6 bits/valeur)
  5. Generation de texte
  6. Affichage des metriques de compression

Usage :
    python scripts/demo_reel.py
    python scripts/demo_reel.py --steps 50 --profile tiny
    python scripts/demo_reel.py --steps 200 --profile base  --data "mon_texte.txt"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ajout du chemin src pour l'import
_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import torch

from ternair.model.modeling import TernairForCausalLM
from ternair.model.generation import generate
from ternair.model.size_profiles import tiny_profile, base_profile
from ternair.training.trainer import train_one_step
from ternair.training.data import (
    CharTokenizer,
    tokenise_corpus,
    toy_corpus,
)
from ternair.quantization.ternary import stats_from
from ternair.quantization.linear import TernairLinear


PROFILES = {
    "tiny": tiny_profile,
    "base": base_profile,
}


def load_custom_text(path: str) -> str:
    """Charge un fichier texte comme corpus d'entrainement."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def collect_ternary_stats(model: TernairForCausalLM) -> None:
    """Analyse les poids ternaires du modele."""
    total_params = 0
    total_pos = 0
    total_neg = 0
    total_zero = 0
    total_gamma = 0.0
    count_gamma = 0

    print("\n  [Analyse des poids ternaires]")
    print(f"  {'Couche':<30} {'Parametres':>10} {'Positifs':>8} {'Negatifs':>8} {'Zeros':>8} {'Sparsite':>8} {'Gamma':>8}")
    print("  " + "-" * 80)

    for name, module in model.named_modules():
        if isinstance(module, TernairLinear):
            w_t, gamma = module.ternarize_parameter()
            stats = stats_from(w_t, gamma)
            total_params += stats.numel
            total_pos += stats.num_pos
            total_neg += stats.num_neg
            total_zero += stats.num_zero
            total_gamma += stats.gamma
            count_gamma += 1

            print(
                f"  {name:<30} {stats.numel:>10,} "
                f"{stats.num_pos:>8,} {stats.num_neg:>8,} {stats.num_zero:>8,} "
                f"{stats.sparsity:>7.1%} {stats.gamma:>7.4f}"
            )

    if count_gamma > 0:
        print("  " + "-" * 80)
        print(
            f"  {'TOTAL':<30} {total_params:>10,} "
            f"{total_pos:>8,} {total_neg:>8,} {total_zero:>8,} "
            f"{total_zero / total_params:>7.1%} {total_gamma / count_gamma:>7.4f}"
        )


def show_size_breakdown(model: TernairForCausalLM) -> None:
    """Affiche le poids memoire du modele."""
    print("\n  [Stockage memoire]")

    # Taille avant gel (poids FP32)
    fp32_bytes = sum(p.numel() * 4 for p in model.parameters())
    fp32_mib = fp32_bytes / (1024**2)

    # Taille apres gel (ternaire compacte)
    model.freeze_storage()
    packed_bytes = model.num_bytes(embedding_dtype_bytes=2)  # FP16 pour embedding
    packed_mib = packed_bytes / (1024**2)
    total_params = model.count_parameters()

    print(f"  Parametres ternaires          : {total_params:,}")
    print(f"  Poids FP32 (avant gel)        : {fp32_mib:.1f} Mio")
    print(f"  Poids ternaire compacte        : {packed_mib:.1f} Mio")
    print(f"  Compression                    : {fp32_mib / packed_mib:.1f}x")
    print(f"  Equivalent bits/parametre      : {packed_bytes * 8 / total_params:.2f} bits")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ternair - Demo reel")
    parser.add_argument("--steps", type=int, default=20, help="Nombre d'etapes d'entrainement")
    parser.add_argument("--lr", type=float, default=3e-4, help="Taux d'apprentissage")
    parser.add_argument("--profile", choices=list(PROFILES), default="tiny", help="Profil du modele")
    parser.add_argument("--storage", choices=["packed", "int8"], default="packed", help="Mode de stockage")
    parser.add_argument("--data", type=str, default=None, help="Chemin vers un fichier texte personnalise")
    parser.add_argument("--prompt", type=str, default="hello world", help="Texte d'amorce pour la generation")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Nombre de tokens a generer")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Chargement des donnees
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  TERNAIR - DEMO REELLE")
    print("  Entrainement et generation avec poids ternaires")
    print("=" * 70)

    if args.data:
        print(f"\n[1/6] Chargement du fichier : {args.data}")
        corpus = load_custom_text(args.data)
    else:
        print("\n[1/6] Utilisation du corpus integre (6 phrases)")
        corpus = toy_corpus()

    print(f"  Corpus : {len(corpus)} caracteres, {len(corpus.split())} mots")

    tok = CharTokenizer(corpus)
    print(f"  Vocabulaire : {tok.vocab_size} tokens (caracteres)")

    # ------------------------------------------------------------------
    # 2. Construction du modele
    # ------------------------------------------------------------------
    print(f"\n[2/6] Construction du modele (profil={args.profile}, stockage={args.storage})")
    profile_fn = PROFILES[args.profile]
    config = profile_fn(storage=args.storage)
    print(f"  Couches : {config.num_hidden_layers}")
    print(f"  Dimension cachee : {config.hidden_size}")
    print(f"  Tetes d'attention : {config.num_attention_heads} (GQA: {config.num_key_value_heads} KV)")

    model = TernairForCausalLM(config)
    total = model.count_parameters()
    print(f"  Parametres ternaires : {total:,} ({total / 1e6:.1f}M)")

    # ------------------------------------------------------------------
    # 3. Entrainement
    # ------------------------------------------------------------------
    print(f"\n[3/6] Entrainement ({args.steps} etapes, lr={args.lr})")

    ids, _ = tokenise_corpus(corpus)
    print(f"  Sequence d'entrainement : {ids.shape[1]} tokens")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = time.time()

    for step in range(1, args.steps + 1):
        loss, _ = train_one_step(model, ids, optimizer=optimizer)
        if step % 5 == 0 or step == 1 or step == args.steps:
            print(f"  Etape {step:>4d}/{args.steps}  loss={loss:.4f}")

    elapsed = time.time() - start
    print(f"  Entrainement termine en {elapsed:.1f}s ({elapsed / max(args.steps, 1):.2f}s/step)")

    # ------------------------------------------------------------------
    # 4. Analyse des poids ternaires
    # ------------------------------------------------------------------
    print(f"\n[4/6] Analyse de la quantification ternaire")
    collect_ternary_stats(model)

    # ------------------------------------------------------------------
    # 5. Gel et generation
    # ------------------------------------------------------------------
    print(f"\n[5/6] Gel du stockage ternaire et generation")
    model.eval()

    print(f"  Taille avant gel : {sum(p.numel() * 4 for p in model.parameters()) / 1024**2:.1f} Mio (FP32)")

    model.freeze_storage()
    packed_bytes = model.num_bytes(embedding_dtype_bytes=2)
    print(f"  Taille apres gel  : {packed_bytes / 1024**2:.1f} Mio (ternaire compacte)")

    # Generation
    prompt_ids = torch.tensor(
        [tok.bos_id] + tok.encode(args.prompt), dtype=torch.long
    ).unsqueeze(0)

    print(f"\n  Amorce      : {args.prompt!r}")
    generated = generate(
        model, prompt_ids, max_new_tokens=args.max_new_tokens, eos_token_id=tok.eos_id
    )
    generated_text = tok.decode(generated[0].tolist())
    print(f"  Genere      : {generated_text!r}")
    print(f"  Tokens genes : {args.max_new_tokens}")

    # ------------------------------------------------------------------
    # 6. Bilan
    # ------------------------------------------------------------------
    print(f"\n[6/6] Bilan")
    print(f"  Modele         : Ternair {args.profile} ({total:,} parametres)")
    print(f"  Stockage       : {packed_bytes / 1024**2:.1f} Mio (compacte)")
    print(f"  Compression    : {sum(p.numel() * 4 for p in model.parameters()) / packed_bytes:.1f}x vs FP32")
    print(f"  Perte finale   : {loss:.4f}")
    print(f"  Generation     : {generated_text!r}")
    print("=" * 70)


if __name__ == "__main__":
    main()
