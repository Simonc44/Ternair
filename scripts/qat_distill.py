#!/usr/bin/env python3
"""
Wordy-QAT : Distillation + Quantization-Aware Training de Ternair

Pipeline :
  1. Charger un modele professeur ultra-performant (SmolLM2-360M, Qwen2.5-0.5B)
  2. Convertir ses nn.Linear en TernairLinear (preservation des poids)
  3. Distillation KL (prof freeze, eleve train) sur FineWeb-Edu
  4. Alpha appris par canal (facteur d'echelle entrainable)
  5. Mixed-precision : Embeddings/LM Head restent en FP16
  6. Gel en stockage compact et export sur HuggingFace Hub

Auteur : Simonc44
Licence : Apache 2.0
"""

# ============================================================================
# CELLULE 1 - Installation
# ============================================================================

import subprocess, sys, math, torch, os
from pathlib import Path

print("=" * 70)
print("  WORDY-QAT : Distillation Ternaire 1.58-bit")
print("=" * 70)

subprocess.run([sys.executable, "-m", "pip", "install",
    "--upgrade", "pip", "setuptools", "wheel"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install",
    "git+https://github.com/Simonc44/Ternair.git",
    "datasets", "accelerate", "huggingface_hub", "pyyaml",
    "transformers", "accelerate"], check=True)

print("Installation terminee")

# ============================================================================
# CELLULE 2 - Connexion HuggingFace + Configuration
# ============================================================================

from huggingface_hub import notebook_login, whoami, create_repo, HfApi

notebook_login()
USERNAME = whoami()["name"]
REPO_ID = f"{USERNAME}/Wordy-QAT"

# Configuration
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"   # Modele professeur
# Alternatives : "Qwen/Qwen2.5-0.5B", "TinyLlama/TinyLlama-1.1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

TRAIN_STEPS = 2000        # 1000-5000 steps recommande
BATCH_SIZE = 2
SEQ_LENGTH = 512
LR = 2e-4
WARMUP_STEPS = 200
TEMPERATURE = 4.0         # Temperature de distillation
ALPHA_KL = 0.5            # Poids KL (0.5 = equilibre CE/KL)

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_SUBSET = "sample-100BT"
DATASET_SAMPLES = 5000

print(f"\nConfiguration :")
print(f"  Modele professeur : {MODEL_NAME}")
print(f"  Device            : {DEVICE}")
print(f"  Steps             : {TRAIN_STEPS}")
print(f"  Batch size        : {BATCH_SIZE}")
print(f"  Sequence length   : {SEQ_LENGTH}")
print(f"  Temperature       : {TEMPERATURE}")
print(f"  Dataset           : {DATASET_NAME}")

# ============================================================================
# CELLULE 3 - Chargement du professeur et du dataset
# ============================================================================

print("\n" + "=" * 70)
print("  CHARGEMENT DU PROFESSEUR")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"Chargement de {MODEL_NAME} en FP16 ...")
teacher = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
).to(DEVICE)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

n_params = sum(p.numel() for p in teacher.parameters())
print(f"Prof charge : {n_params:,} parametres")

# Charger le dataset
print(f"\nChargement du dataset {DATASET_NAME} ...")
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset(
    DATASET_NAME, DATASET_SUBSET,
    split=f"train[:{DATASET_SAMPLES}]",
    streaming=False,
)

def tokenize_fn(examples):
    texts = examples.get("text", [""])
    all_ids = tokenizer(
        texts, truncation=True, max_length=SEQ_LENGTH,
        padding=False, return_tensors=None,
    )["input_ids"]
    return {"input_ids": all_ids}

dataset = dataset.map(tokenize_fn, batched=True, batch_size=100)
dataset = dataset.filter(lambda x: len(x["input_ids"]) == SEQ_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Dataset pret : {len(dataset)} echantillons, {len(dataloader)} batches")

# ============================================================================
# CELLULE 4 - Conversion du professeur en eleve ternaire
# ============================================================================

print("\n" + "=" * 70)
print("  CONVERSION -> TERNAIRLINEAR (preservation des poids)")
print("=" * 70)

import copy
from ternair.quantization.distillation import (
    convert_model_to_ternair,
    count_ternary_params,
    distillation_loss,
)

student = copy.deepcopy(teacher)
student = convert_model_to_ternair(
    student,
    storage="packed",
    keep_embed_fp16=True,       # Mixed-precision : embedding/LM head en FP16
    learned_alpha=True,          # Alpha appris par canal
)
student = student.to(DEVICE)
student.train()

stats = count_ternary_params(student)
print(f"\n  Poids ternaires : {stats['ternary_params']:,} ({stats['ternary_ratio']:.1%})")
print(f"  Poids FP16      : {stats['fp16_params']:,}")
print(f"  Total           : {stats['total']:,}")

# ============================================================================
# CELLULE 5 - DISTILLATION QAT
# ============================================================================

print("\n" + "=" * 70)
print("  DISTILLATION QAT")
print("=" * 70)

optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.02)

# Warmup + cosine decay
def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, TRAIN_STEPS - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

global_step = 0
best_loss = float("inf")

for epoch in range(10):
    student.train()
    epoch_loss = 0.0
    
    for batch in dataloader:
        if global_step >= TRAIN_STEPS:
            break
        
        input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze(1)
        
        # Forward professeur (freeze -> pas de gradients)
        with torch.no_grad():
            teacher_logits = teacher(input_ids)
        
        # Forward eleve (avec TernairLinear + alpha appris)
        student_logits = student(input_ids)
        
        # Perte combinee CE + KL
        labels = input_ids
        loss = distillation_loss(
            student_logits, teacher_logits, labels,
            alpha=ALPHA_KL,
            temperature=TEMPERATURE,
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()
        global_step += 1
        
        current_lr = scheduler.get_last_lr()[0]
        if global_step % 10 == 0:
            print(f"  Step {global_step:>5d}/{TRAIN_STEPS}  "
                  f"loss={loss.item():.4f}  lr={current_lr:.2e}")
    
    avg_loss = epoch_loss / max(len(dataloader), 1)
    print(f"\n  >>> Epoque {epoch+1}  loss moyenne={avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        print(f"  >>> Nouveau meilleur score !")

print(f"\nDistillation terminee en {global_step} etapes. Meilleure perte : {best_loss:.4f}")

# ============================================================================
# CELLULE 6 - GEL + EXPORT
# ============================================================================

print("\n" + "=" * 70)
print("  GEL DU STOCKAGE TERNAIRE")
print("=" * 70)

student.eval()
student.cpu()

# Gel des couches TernairLinear
from ternair.quantization.linear import TernairLinear
for module in student.modules():
    if isinstance(module, TernairLinear):
        module.freeze_storage()

# Stats memoire
ternary_size = 0
for module in student.modules():
    if isinstance(module, TernairLinear):
        ternary_size += module.state_bytes()

total_size = ternary_size + sum(p.numel() * 2 for n, p in student.named_parameters()
    if not any(m in n for m in ["TernairLinear", "packed_weight", "gamma_eval"]))

print(f"  Taille ternaire compactee : {ternary_size / 1024**2:.1f} Mio")
print(f"  Taille totale (FP16+ternaire) : {total_size / 1024**2:.1f} Mio")
print(f"  Compression vs FP32 : "
      f"{sum(p.numel() * 4 for p in teacher.parameters()) / total_size:.1f}x")

# ============================================================================
# CELLULE 7 - TEST DE GENERATION
# ============================================================================

print("\n" + "=" * 70)
print("  TEST DE GENERATION")
print("=" * 70)

from ternair.model.generation import generate

prompts = [
    "The future of AI is",
    "Explain quantum computing in simple terms:",
    "Write a short poem about neural networks:",
]

for prompt in prompts:
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    out = generate(
        student, prompt_ids,
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    print(f"\n  [Prompt] {prompt}")
    print(f"  [Wordy]  {text}")

# ============================================================================
# CELLULE 8 - TELEVERSEMENT SUR HUGGINGFACE HUB
# ============================================================================

print("\n" + "=" * 70)
print("  TELEVERSEMENT SUR HUGGINGFACE HUB")
print("=" * 70)

# Creer le depot
create_repo(REPO_ID, repo_type="model", exist_ok=True)

# Info d'entrainement
training_info = {
    "teacher_model": MODEL_NAME,
    "steps": TRAIN_STEPS,
    "batch_size": BATCH_SIZE,
    "seq_length": SEQ_LENGTH,
    "lr": LR,
    "temperature": TEMPERATURE,
    "alpha_kl": ALPHA_KL,
    "dataset": DATASET_NAME,
    "final_loss": best_loss,
    "ternary_params": stats['ternary_params'],
    "size_mib": total_size / 1024**2,
    "method": "QAT + distillation + alpha appris + mixed-precision",
}

# Sauvegarder le modele
torch.save({
    "config": {
        "teacher": MODEL_NAME,
        "ternair_config": {
            "storage": "packed",
            "keep_embed_fp16": True,
            "learned_alpha": True,
        },
        "training": training_info,
    },
    "state_dict": student.state_dict(),
}, "wordy_qat_model.pt")
print("Modele sauvegarde : wordy_qat_model.pt")

# Carte du modele
card = f"""---
language: en
license: apache-2.0
library_name: ternair
tags:
- ternary
- bitnet
- b1.58
- qat
- distillation
datasets:
- {DATASET_NAME}
---

# Wordy-QAT : Modele Ternaire 1.58-bit par Distillation

## Description

Modele de langue ternaire (BitNet b1.58) obtenu par distillation de
{MODEL_NAME}. Les poids originaux du professeur sont preserves et
quantifies en ternaire via TernairLinear avec alpha appris par canal.

## Details

| Propriete | Valeur |
|-----------|--------|
| Professeur | {MODEL_NAME} |
| Methode | QAT + Distillation KL |
| Poids ternaires | {stats['ternary_params']:,} |
| Poids FP16 | {stats['fp16_params']:,} |
| Taille compactee | {total_size / 1024**2:.1f} Mio |
| Etapes | {TRAIN_STEPS} |
| Dataset | {DATASET_NAME} |
| Perte finale | {best_loss:.4f} |

## Utilisation

```python
import torch
from ternair.model.modeling import TernairForCausalLM
from ternair.model.config import TernairConfig
from ternair.model.generation import generate
from transformers import AutoTokenizer

checkpoint = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/{REPO_ID}/resolve/main/wordy_qat_model.pt",
    map_location="cpu",
)
student_state = checkpoint["state_dict"]

tokenizer = AutoTokenizer.from_pretrained("{MODEL_NAME}")

# Inference sans Ternair : charger le modele original et appliquer les poids
# (Voir le guide complet pour l'inference Ternair native)
prompt = tokenizer.encode("The future of AI is", return_tensors="pt")
# ...
```

## Entrainement

Distillation QAT avec alpha appris et mixed-precision selective
(embedding/LM head en FP16, reste en ternaire 1.58-bit).
"""

with open("README.md", "w") as f:
    f.write(card)

# Upload
api = HfApi()
api.upload_file(
    path_or_fileobj="wordy_qat_model.pt",
    path_in_repo="wordy_qat_model.pt",
    repo_id=REPO_ID,
)
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
)

print(f"\n  >>> https://huggingface.co/{REPO_ID}")
print("=" * 70)
