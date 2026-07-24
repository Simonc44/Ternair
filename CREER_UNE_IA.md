# Creer une IA avec Ternair

Ce guide vous montre comment creer, entrainer et utiliser un modele de langue ternaire (BitNet b1.58) avec la librairie Ternair. Pas a pas, comme dans le script `scripts/demo_reel.py`.

---

## 1. Installation

```bash
# Cloner le depot
git clone https://github.com/Simonc44/Ternair.git
cd Ternair

# Environnement Python
python3 -m venv .venv
source .venv/bin/activate

# Installer Ternair
pip install --upgrade setuptools wheel
pip install -e .
```

## 2. Principe general

Ternair cree un modele de langue **des la base** avec des poids ternaires (`{-1, 0, +1}`). Cela permet une compression ~16x par rapport au FP16.

Le cycle de vie d'un modele Ternair :

```
Donnees texte
    |
    v
Construire le modele (TernairForCausalLM)
    |
    v
Entrainer (AdamW + STE)
    |
    v
Geler le stockage (freeze_storage -> 1,6 bits/valeur)
    |
    v
Generer du texte (generate)
    |
    v
Exporter / Distribuer
```

## 3. Modele pret a l'emploi (le plus simple)

```python
from pathlib import Path
import sys

# Ajouter src/ au chemin
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from ternair.model.modeling import TernairForCausalLM
from ternair.model.generation import generate
from ternair.model.size_profiles import tiny_profile, base_profile, one_gb_profile
from ternair.training.data import CharTokenizer, tokenise_corpus, toy_corpus

# --- 1. Choisir un profil ---
# Profils disponibles :
#   tiny   -> 2,6 M parametres  (~6 Mio)
#   base   -> 700 M parametres  (~150 Mio)
#   one_gb -> 4,07 B parametres (~942 Mio)
config = tiny_profile(storage="packed")

# --- 2. Construire le modele ---
model = TernairForCausalLM(config)
print(f"Parametres : {model.count_parameters():,}")

# --- 3. Preparer les donnees ---
# Corpus integre (6 phrases)
corpus = toy_corpus()

# Ou bien chargez votre propre fichier texte :
# with open("mon_texte.txt", "r") as f:
#     corpus = f.read()

# Tokeniseur character-level
tok = CharTokenizer(corpus)
ids, _ = tokenise_corpus(corpus)
print(f"Tokens d'entrainement : {ids.shape[1]}")

# --- 4. Entrainer ---
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(20):
    logits = model(ids)

    # Cross-entropy next-token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = ids[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 5 == 0:
        print(f"Etape {step}  loss={loss.item():.4f}")

# --- 5. Geler en stockage ternaire compact ---
model.eval()
snapshot = model.freeze_storage()
print(f"Taille compactee : {model.num_bytes() / 1024**2:.1f} Mio")

# --- 6. Generer du texte ---
prompt = torch.tensor(
    [tok.bos_id] + tok.encode("hello world"), dtype=torch.long
).unsqueeze(0)

out = generate(model, prompt, max_new_tokens=16, eos_token_id=tok.eos_id)
print(f"Genere : {tok.decode(out[0].tolist())!r}")
```

## 4. Avec des vraies donnees (HuggingFace datasets)

```python
# Installation
# pip install datasets accelerate pyyaml

# Entrainement via le script fourni :
accelerate launch scripts/train.py --config scripts/train_tiny.yaml

# Pour un entrainement personnalise :
from ternair.training.data import build_dataloader
from ternair.training.config import TrainingConfig
from ternair.training.trainer import build_model, train_one_epoch

cfg = TrainingConfig(
    model_profile="tiny",
    model_storage="packed",
    dataset_name="HuggingFaceFW/fineweb-edu",
    dataset_subset="sample-100BT",
    dataset_streaming=True,
    dataset_max_samples=1000,
    tokenizer_name="gpt2",
    batch_size=2,
    seq_length=128,
    max_train_steps=100,
)

model = build_model(cfg)
dataloader = build_dataloader(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Entrainement sur CPU (sans accelerate)
model.train()
for step, batch in enumerate(dataloader):
    if step >= cfg.max_train_steps:
        break
    input_ids = batch["input_ids"]
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
        input_ids[..., 1:].contiguous().view(-1),
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if step % 10 == 0:
        print(f"Etape {step}  loss={loss.item():.4f}")
```

## 5. Analyser les poids ternaires

```python
from ternair.quantization.ternary import stats_from, ternarize

# Examiner une couche
for name, module in model.named_modules():
    if hasattr(module, "ternarize_parameter"):
        w_t, gamma = module.ternarize_parameter()
        stats = stats_from(w_t, gamma)
        print(f"{name} : "
              f"{stats.num_pos / stats.numel:>5.1%} [+1]  "
              f"{stats.num_zero / stats.numel:>5.1%} [0]  "
              f"{stats.num_neg / stats.numel:>5.1%} [-1]  "
              f"gamma={stats.gamma:.4f}")
```

## 6. Exporter et charger un modele

```python
# --- Sauvegarder ---
model.freeze_storage()
model.eval()
torch.save({
    "config": model.config.to_dict(),
    "state_dict": model.state_dict(),
}, "mon_modele_ternaire.pt")

# --- Charger ---
checkpoint = torch.load("mon_modele_ternaire.pt")
config = TernairConfig(**checkpoint["config"])
model = TernairForCausalLM(config)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
```

## 7. Profils disponibles

| Profil | Parametres | Taille compactee | Usage |
|--------|-----------|-----------------|-------|
| `tiny_profile()` | 2,6 M | ~2,5 Mio | Tests, debug |
| `base_profile()` | 700 M | ~150 Mio | Entrainement local |
| `one_gb_profile()` | 4,07 B | ~942 Mio | Production, < 1 Gio |

Personnalisation :

```python
from ternair.model.config import TernairConfig

mon_config = TernairConfig(
    vocab_size=32000,
    hidden_size=1024,           # Dimension cachee
    intermediate_size=2816,     # Dimension MLP
    num_hidden_layers=12,       # Nombre de couches
    num_attention_heads=16,     # Tetes d'attention
    num_key_value_heads=4,      # GQA (4 KV tetes)
    max_position_embeddings=2048,
    storage="packed",           # "packed" | "int8" | "fastpacked"
)
model = TernairForCausalLM(mon_config)
```

## 8. Commandes CLI rapides

```bash
# Voir la configuration du modele
python -m ternair info --profile one_gb

# Projection de taille
python -m ternair size --profile one_gb

# Demo complete (construction + entrainement + generation)
PYTHONPATH=src python3 scripts/demo_reel.py --steps 20

# Avec vos donnees
PYTHONPATH=src python3 scripts/demo_reel.py --steps 50 --data mon_fichier.txt

# Entrainement longue duree
accelerate launch scripts/train.py --config scripts/train_tiny.yaml
```

## 9. Structure du projet pour les developpeurs

```
Ternair/
├── src/ternair/          # Code source de la librairie
│   ├── quantization/     # Quantification ternaire (STE, packing)
│   ├── model/            # Architecture du modele (attention, MLP, SSM)
│   ├── kernels/          # Noyaux CPU/GPU optimises
│   ├── training/         # Pipeline d'entrainement
│   └── benchmark/        # Calcul de taille et performances
├── scripts/
│   ├── demo_reel.py      # Demo cle en main (ce guide en code)
│   └── train.py          # Entrainement avec accelerate
└── tests/                # Tests
```

## 10. Points cle a retenir

1. **Pas de modele pre-entraine** : Ternair entraine depuis zero. Pas de chargement de Llama ou GPT-2.
2. **Compression 16x** : 4 milliards de parametres dans 942 Mio.
3. **STE** : La quantification est differentielle -> on peut entrainer normalement.
4. **Gel obligatoire** : Appelez `freeze_storage()` avant l'inference pour activer le stockage compact.
5. **CPU only** : Les profils tiny et base tournent sur CPU.
6. **Pas de cache KV** : La generation est en O(n^2) mais l'empreinte memoire est minimale.
