#!/usr/bin/env python3
"""
Wordy - Creer une IA puissante avec Ternair sur Google Colab

Execute ce script directement sur Google Colab (Runtime -> Run all).
Il cree un modele ternaire BitNet b1.58, l'entraine sur un dataset
HuggingFace, et le televerse sur HuggingFace Hub.

Auteur : Simonc44
Licence : Apache 2.0
"""

# ============================================================================
# CELLULE 1 - Installation de Ternair et dependances
# ============================================================================

import subprocess
import sys
import os
import math
from pathlib import Path

print("=" * 70)
print("  WORDY - Installation des dependances")
print("=" * 70)

# Installer Ternair depuis GitHub (derniere version avec les corrections)
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "--upgrade", "pip", "setuptools", "wheel"
], check=True)

subprocess.run([
    sys.executable, "-m", "pip", "install",
    "git+https://github.com/Simonc44/Ternair.git",
], check=True)

# Dependances supplementaires
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "datasets", "accelerate", "huggingface_hub", "pyyaml"
], check=True)

print("Installation terminee.")

# ============================================================================
# CELLULE 2 - Connexion HuggingFace Hub (televersement du modele)
# ============================================================================

from huggingface_hub import notebook_login, whoami, HfApi, create_repo

print("\n" + "=" * 70)
print("  WORDY - Connexion HuggingFace Hub")
print("=" * 70)
print("\nConnecte-toi avec ton token HuggingFace (https://huggingface.co/settings/tokens)")
notebook_login()

USERNAME = whoami()["name"]
REPO_NAME = "Wordy-Ternair"
REPO_ID = f"{USERNAME}/{REPO_NAME}"

print(f"Compte : {USERNAME}")
print(f"Depot  : {REPO_ID}")

# ============================================================================
# CELLULE 3 - Configuration du modele Wordy
# ============================================================================

print("\n" + "=" * 70)
print("  WORDY - Configuration du modele")
print("=" * 70)

import torch
import numpy as np
from datasets import load_dataset

# Choix du profil
# - "tiny"   : 2,6M params, tourne sur CPU/GPU gratuit (recommandé pour test)
# - "base"   : 700M params, nécessite GPU T4 (Colab gratuit)
# - "one_gb" : 4B params, nécessite GPU A100 (Colab Pro)
PROFILE_NAME = "tiny"       # "tiny" | "base" | "one_gb"
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"

print(f"Profil       : {PROFILE_NAME}")
print(f"GPU disponible : {USE_GPU}")
print(f"Device        : {DEVICE}")

# Configuration selon le profil
if PROFILE_NAME == "tiny":
    from ternair.model.size_profiles import tiny_profile as profile_fn
    TRAIN_STEPS = 2000       # 2000 etapes x 4 batch x 128 seq = 1M tokens vus
    BATCH_SIZE = 4
    SEQ_LENGTH = 128
    DATASET_SAMPLES = 5000
    WARMUP_STEPS = 100
elif PROFILE_NAME == "base":
    from ternair.model.size_profiles import base_profile as profile_fn
    TRAIN_STEPS = 50000      # 50K etapes pour convergence sérieuse
    BATCH_SIZE = 2
    SEQ_LENGTH = 256
    DATASET_SAMPLES = 50000
    WARMUP_STEPS = 1000
elif PROFILE_NAME == "one_gb":
    from ternair.model.size_profiles import one_gb_profile as profile_fn
    TRAIN_STEPS = 100000     # 100K etapes
    BATCH_SIZE = 1
    SEQ_LENGTH = 512
    DATASET_SAMPLES = 100000
    WARMUP_STEPS = 2000
else:
    raise ValueError(f"Profil inconnu : {PROFILE_NAME}")

LR = 3e-4
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_SUBSET = "sample-100BT"

config = profile_fn(storage="packed")
print(f"\nConfiguration du modele :")
print(f"  Couches          : {config.num_hidden_layers}")
print(f"  Dimension cachee  : {config.hidden_size}")
print(f"  Tetes d'attention : {config.num_attention_heads}")
print(f"  Parametres        : {config.num_hidden_layers * config.hidden_size * config.intermediate_size:,}")

# ============================================================================
# CELLULE 4 - Chargement du dataset HuggingFace
# ============================================================================

print("\n" + "=" * 70)
print("  WORDY - Chargement du dataset HuggingFace")
print("=" * 70)

print(f"Dataset : {DATASET_NAME}/{DATASET_SUBSET}")
print(f"Echantillons : {DATASET_SAMPLES}")

dataset = load_dataset(
    DATASET_NAME, DATASET_SUBSET,
    split=f"train[:{DATASET_SAMPLES}]",
    streaming=False,
)

print(f"Dataset charge : {len(dataset)} echantillons")

# Tokenizer GPT-2
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    texts = examples.get("text", examples.get("content", [""]))
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text, truncation=False)
        if len(ids) == 0:
            ids = [tokenizer.eos_token_id or 0]
        all_ids.extend(ids + [tokenizer.eos_token_id or 0])

    input_ids = []
    for i in range(0, len(all_ids), SEQ_LENGTH):
        chunk = all_ids[i : i + SEQ_LENGTH]
        if len(chunk) == SEQ_LENGTH:
            input_ids.append(chunk)
    return {"input_ids": input_ids}

print("Tokenisation en cours...")
dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=100,
    remove_columns=dataset.column_names,
)

print(f"Echantillons tokenises : {len(dataset)}")
print(f"Longueur sequence : {SEQ_LENGTH} tokens")

from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

print(f"Nombre de batches : {len(dataloader)}")

# ============================================================================
# CELLULE 5 - Construction du modele Wordy
# ============================================================================

print("\n" + "=" * 70)
print("  WORDY - Construction du modele")
print("=" * 70)

from ternair.model.modeling import TernairForCausalLM

model = TernairForCausalLM(config)
model = model.to(DEVICE)

total_params = model.count_parameters()
print(f"Parametres ternaires : {total_params:,} ({total_params / 1e6:.1f}M)")
print(f"Modele envoye sur {DEVICE}")

# Learning rate scheduler avec warmup + cosine decay
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)  # warmup lineaire
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay
    return LambdaLR(optimizer, lr_lambda)

# ============================================================================
# CELLULE 6 - Entrainement
# ============================================================================

print("\n" + "=" * 70)
print("  WORDY - Entrainement")
print("=" * 70)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, TRAIN_STEPS)
global_step = 0

for epoch in range(max(1, TRAIN_STEPS // len(dataloader) + 1)):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        if global_step >= TRAIN_STEPS:
            break

        input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze(1)

        logits = model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = input_ids[..., 1:].contiguous()

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        num_batches += 1
        global_step += 1

        current_lr = scheduler.get_last_lr()[0]
        if global_step % 10 == 0:
            print(f"  Etape {global_step:>5d}/{TRAIN_STEPS}  loss={loss.item():.4f}  lr={current_lr:.2e}")

    avg_loss = epoch_loss / max(num_batches, 1)
    print(f"  >>> Epoque {epoch+1}  loss moyenne={avg_loss:.4f}")

print(f"\nEntrainement termine en {global_step} etapes.")
print(f"Perte finale : {loss.item():.4f}")

# ============================================================================
# CELLULE 7 - Gel en stockage ternaire compact
# ============================================================================

print("\n" + "=" * 70)
print("  WORDY - Gel du stockage ternaire")
print("=" * 70)

model.eval()
model.cpu()

# Taille avant gel
fp32_size = sum(p.numel() * 4 for p in model.parameters()) / (1024**3)
print(f"Taille FP32 (avant gel) : {fp32_size:.2f} Gio")

# Gel en stockage compact
model.freeze_storage()

# Taille apres gel
packed_size = model.num_bytes(embedding_dtype_bytes=2) / (1024**3)
total_params = model.count_parameters()
bits_per_param = model.num_bytes(embedding_dtype_bytes=2) * 8 / total_params

print(f"Taille compacte (apres gel) : {packed_size:.2f} Gio")
print(f"Compression vs FP32 : {fp32_size / packed_size:.1f}x")
print(f"Bits par parametre : {bits_per_param:.2f}")

# ============================================================================
# CELLULE 8 - Test de generation
# ============================================================================

print("\n" + "=" * 70)
print("  WORDY - Test de generation")
print("=" * 70)

from ternair.model.generation import generate

prompts = [
    "The future of AI is",
    "In the beginning,",
    "Machine learning enables",
    "Once upon a time,",
]

for prompt in prompts:
    # Tokenizer le prompt
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generation avec temperature + top-p sampling (texte plus varie)
    out = generate(
        model, prompt_ids,
        max_new_tokens=32,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    print(f"\n  Prompt : {prompt}")
    print(f"  Genere  : {generated_text}")

# ============================================================================
# CELLULE 9 - Televersement sur HuggingFace Hub
# ============================================================================

print("\n" + "=" * 70)
print("  WORDY - Televersement sur HuggingFace Hub")
print("=" * 70)

import json
from datetime import datetime

# Creer le depot
try:
    create_repo(REPO_ID, repo_type="model", exist_ok=True)
    print(f"Depot cree : {REPO_ID}")
except Exception as e:
    print(f"Depot existe deja : {e}")

# Sauvegarder le modele
MODEL_FILE = "wordy_model.pt"
torch.save({
    "config": config.to_dict(),
    "state_dict": model.state_dict(),
    "training_info": {
        "steps": TRAIN_STEPS,
        "dataset": DATASET_NAME,
        "profile": PROFILE_NAME,
        "final_loss": loss.item(),
        "total_params": total_params,
        "size_gib": packed_size,
        "bits_per_param": bits_per_param,
        "date": datetime.now().isoformat(),
    },
}, MODEL_FILE)
print(f"Modele sauvegarde : {MODEL_FILE}")

# Carte du modele (README)
card_content = f"""---
language: en
license: apache-2.0
library_name: ternair
tags:
- ternary
- bitnet
- b1.58
- efficient-ai
datasets:
- {DATASET_NAME}
---

# Wordy - Ternair

Un modele de langue ternaire (BitNet b1.58) entraine avec Ternair.

## Details

| Propriete | Valeur |
|-----------|--------|
| Profil | {PROFILE_NAME} |
| Parametres | {total_params:,} |
| Taille compactee | {packed_size:.3f} Gio |
| Bits/parametre | {bits_per_param:.2f} |
| Etapes d'entrainement | {TRAIN_STEPS} |
| Dataset | {DATASET_NAME} |
| Perte finale | {loss.item():.4f} |

## Utilisation

```python
import torch
from ternair.model.modeling import TernairForCausalLM
from ternair.model.config import TernairConfig
from ternair.model.generation import generate

# Charger le modele
checkpoint = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/{REPO_ID}/resolve/main/wordy_model.pt",
    map_location="cpu",
)

config = TernairConfig(**checkpoint["config"])
model = TernairForCausalLM(config)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Generer du texte
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
prompt = tokenizer.encode("The future of AI is", return_tensors="pt")
out = generate(model, prompt, max_new_tokens=32)
print(tokenizer.decode(out[0].tolist(), skip_special_tokens=True))
```

## Entrainement

Ce modele a ete entraine avec Ternair sur {DATASET_NAME}.
"""

with open("README.md", "w") as f:
    f.write(card_content)

# Televerser
api = HfApi()

api.upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo=MODEL_FILE,
    repo_id=REPO_ID,
    repo_type="model",
)
print(f"Modele televerse : {REPO_ID}/{MODEL_FILE}")

api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
)
print(f"Carte televersee : {REPO_ID}/README.md")

print("\n" + "=" * 70)
print(f"  WORDY EST EN LIGNE !")
print(f"  https://huggingface.co/{REPO_ID}")
print("=" * 70)
