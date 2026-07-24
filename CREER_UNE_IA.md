# Creer une IA avec Ternair

Ce guide vous montre comment creer, entrainer et utiliser un modele de langue ternaire (BitNet b1.58) avec la librairie Ternair.

Version couverte : **v0.3.0** (5 nouvelles ameliorations : T-LoRA, OmniQuant, BitAttention, MoE, WebGPU)

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

Ternair cree un modele de langue **des la base** avec des poids ternaires (`{-1, 0, +1}`). Compression ~16x par rapport au FP16.

Le cycle de vie d'un modele Ternair :

```
Donnees texte -> Construire TernairForCausalLM
    -> Entrainer (AdamW + STE + Recuit beta)
    -> Geler (freeze_storage -> 1,6 bits/valeur)
    -> Optionnel : Ajouter adaptateurs T-LoRA
    -> Optionnel : Activer KV-Cache 2-bit
    -> Optionnel : Remplacer MLP par MoE
    -> Generer du texte (generate)
    -> Exporter SafeTensors + Distribuer (Hub HF / WebGPU)
```

## 3. Modele pret a l'emploi (le plus simple)

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from ternair.model.modeling import TernairForCausalLM
from ternair.model.generation import generate
from ternair.model.size_profiles import tiny_profile, base_profile, one_gb_profile
from ternair.training.data import CharTokenizer, tokenise_corpus, toy_corpus

# --- 1. Choisir un profil ---
config = tiny_profile(storage="packed")

# --- 2. Construire le modele ---
model = TernairForCausalLM(config)
print(f"Parametres : {model.count_parameters():,}")

# --- 3. Preparer les donnees ---
corpus = toy_corpus()
tok = CharTokenizer(corpus)
ids, _ = tokenise_corpus(corpus)

# --- 4. Entrainer ---
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(20):
    logits = model(ids)
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

## 4. Specialisation avec T-LoRA (Nouveau v0.3.0)

Ajoutez des adaptateurs ternaires low-rank pour apprendre une nouvelle tache (code, langue, domaine) sans modifier le modele de base.

```python
from ternair.quantization.bitdelta import add_lora_to_model, AdapterRegistry
import torch

# Ajouter des adaptateurs LoRA ternaires (rank=8)
registry = add_lora_to_model(model, rank=8, alpha=1.0)
print(f"Params entrainables: {registry.count_params():,}")
print(f"Soit {registry.count_params() / model.count_parameters():.2%} du modele de base")

# Entrainer seulement les adaptateurs
optimizer = torch.optim.AdamW(registry.adapter_params(), lr=1e-3)

for step in range(100):
    logits = model(ids)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = ids[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Sauvegarder les adaptateurs (quelques Ko seulement)
torch.save(registry.state_dict(), "adaptateurs.pt")

# Recharger plus tard sur un autre modele de base
new_registry = add_lora_to_model(new_model, rank=8)
new_registry.load_state_dict(torch.load("adaptateurs.pt"))
```

## 5. Avec des vraies donnees (HuggingFace datasets)

```python
# pip install datasets accelerate pyyaml

# Entrainement via le script fourni :
accelerate launch scripts/train.py --config scripts/train_tiny.yaml

# Ou personnalise avec FineWeb :
from ternair.training.data import build_dataloader
from ternair.training.config import TrainingConfig
from ternair.training.trainer import build_model

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

## 6. KV-Cache 2-bit pour contextes longs (Nouveau v0.3.0)

Activez le KV-Cache quantifie en 2-bit pour diviser par 4 la memoire du cache d'attention.

```python
from ternair.model.config import TernairConfig

config = TernairConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,
    kv_cache_bits=2,         # 2-bit KV cache (BitAttention)
    max_position_embeddings=32768,  # Contexte ultra-long
)
```

## 7. Experts MoE pour plus de capacite (Nouveau v0.3.0)

Remplacez certains MLP par des blocs a Melange d'Experts. Seulement 2 experts sur 8 sont actifs par token.

```python
from ternair.model.moe import add_moe_to_model

# Convertir certaines couches en MoE
add_moe_to_model(
    model,
    num_experts=8,          # 8 experts ternaires
    top_k=2,               # 2 actifs par token
    moe_layer_period=2,    # Toutes les 2 couches
)
```

## 8. WebGPU -- Inference dans le navigateur (Nouveau v0.3.0)

Exportez votre modele pour le faire tourner directement dans Chrome/Firefox.

```bash
# Generer les shaders WebGPU + runtime JS
python -c "
from ternair.kernels.webternair import generate_wgsl_ternary_matmul, generate_js_runtime

with open('model_shader.wgsl', 'w') as f:
    f.write(generate_wgsl_ternary_matmul())
with open('ternair-web.js', 'w') as f:
    f.write(generate_js_runtime())
print('Fichiers Web generes')
"

# Dans le navigateur :
cat << 'HTML'
<script src="ternair-web.js"></script>
<script>
  const runtime = new TernairWebRuntime();
  await runtime.init();
  await runtime.load('mon_modele.safetensors');
  const result = await runtime.generate([1, 234], 64);
  console.log(result.tokens);
</script>
HTML
```

## 9. Analyser les poids ternaires

```python
from ternair.quantization.ternary import stats_from, ternarize

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

## 10. Exporter et charger un modele

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

# --- Export SafeTensors (HuggingFace) ---
from ternair.model.export import export_to_safetensors, export_huggingface_package

export_to_safetensors(model, "modele.safetensors")
export_huggingface_package(model, output_dir="./mon-modele-hf")
```

## 11. Profils disponibles

| Profil | Parametres | Taille compactee | Usage |
|--------|-----------|-----------------|-------|
| `tiny_profile()` | 2,6 M | ~2,5 Mio | Tests, debug |
| `base_profile()` | 700 M | ~150 Mio | Entrainement local |
| `one_gb_profile()` | 4,07 B | ~942 Mio | Production, < 1 Gio |

Personnalisation avec les nouvelles options v0.3.0 :

```python
from ternair.model.config import TernairConfig

mon_config = TernairConfig(
    vocab_size=32000,
    hidden_size=1024,
    intermediate_size=2816,
    num_hidden_layers=12,
    num_attention_heads=16,
    num_key_value_heads=4,
    max_position_embeddings=2048,
    storage="packed",
    # Options v0.3.0
    kv_cache_bits=2,           # KV cache 2-bit economie memoire
    num_experts=8,             # MoE (si > 1)
    top_k_experts=2,           # Experts actifs par token
    moe_layer_period=4,        # MoE toutes les 4 couches
)
model = TernairForCausalLM(mon_config)
```

## 12. Commandes CLI rapides

```bash
# Voir la configuration
python -m ternair info --profile one_gb

# Projection de taille
python -m ternair size --profile one_gb

# Demo complete
PYTHONPATH=src python3 scripts/demo_reel.py --steps 20

# Tests CI (8 tests v0.3.0)
PYTHONPATH=src python3 scripts/test_ci.py

# Entrainement longue duree
accelerate launch scripts/train.py --config scripts/train_tiny.yaml
```

## 13. Structure du projet

```
Ternair/
├── src/ternair/
│   ├── quantization/
│   │   ├── linear.py         # TernairLinear
│   │   ├── ternary.py        # Ternarisation STE + recuit
│   │   ├── activation.py     # 8-bit + Hadamard + OmniQuant
│   │   ├── bitdelta.py       # T-LoRA adapters (NOUVEAU)
│   │   ├── distillation.py   # KL loss + Feature Matching
│   │   └── packing.py        # Conditionnement compact
│   ├── model/
│   │   ├── config.py         # Configuration
│   │   ├── attention.py      # GQA + KV quant 2-bit
│   │   ├── moe.py            # MoE ternaire (NOUVEAU)
│   │   ├── export.py         # SafeTensors export
│   │   ├── generation.py     # Sampling + streaming + chat
│   │   └── ...
│   ├── kernels/
│   │   ├── inference.h       # Runtime C++ header-only
│   │   ├── webternair.py     # WebGPU backend (NOUVEAU)
│   │   └── ...
│   ├── training/
│   ├── benchmark/
│   └── cli.py
├── scripts/
│   ├── demo_reel.py          # Demo cle en main
│   ├── test_ci.py            # Tests CI (8 tests)
│   └── train.py
```

## 14. Points cle a retenir

1. **Pas de modele pre-entraine** : Ternair entraine depuis zero.
2. **Compression 16x** : 4 milliards de parametres dans 942 Mio.
3. **STE** : La quantification est differentielle -> on peut entrainer normalement.
4. **T-LoRA** : Specialisez sans retoucher le modele de base.
5. **OmniQuant** : Calibrez les echelles pour moins de perte de qualite.
6. **BitAttention** : KV-cache 2-bit pour contextes longs sur petite RAM.
7. **MoE** : Plus de parametres, meme cout de calcul par token.
8. **WebGPU** : Inference dans le navigateur, zero backend.
9. **Gel obligatoire** : `freeze_storage()` avant inference.
