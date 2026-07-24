[![CI](https://github.com/Simonc44/Ternair/actions/workflows/ci.yml/badge.svg)](https://github.com/Simonc44/Ternair/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.4%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/Simonc44/Ternair?include_prereleases)](https://github.com/Simonc44/Ternair/releases)

# Ternair

**BitNet b1.58 -- Reseaux de neurones ternaires a l'echelle 1 Go**

---

Ternair est une implementation de production de **BitNet b1.58**, une architecture de reseau de neurones dans laquelle chaque poids est contraint a `{-1, 0, +1}` (valeurs ternaires, ~1,58 bits). Cette approche permet :

- Compression memoire ~16x par rapport au FP16 (942 Mio pour un modele de 4 milliards de parametres)
- Arithmetique par additions et soustractions uniquement -- zero multiplication flottante durant l'inference
- Absence de cache KV lors de l'utilisation des couches SSM optionnelles (memoire de generation en O(1))
- Compression de tokens K-WTA via le goulot thalamique (ThalamicBottleneck) : 32 latents fixes par sequence

## Fonctionnalites (v0.5.0)

| Fonctionnalite | Statut |
|----------------|--------|
| Quantification ternaire (STE, echelle gamma, 3 valeurs) | Disponible |
| Conditionnement : base-3 (1,6 b/v) ou 2 bits (2,0 b/v) | Disponible |
| Noyau GPU (Triton -- decodage en boucle JIT) | Disponible |
| Noyau CPU (C++ SIMD -- AVX-512 / ARM NEON) | Disponible |
| Modele causal LM (attention GQA, RoPE, MLP SwiGLU ternaire) | Disponible |
| ThalamicBottleneck (compression K-WTA, K=32) | Disponible |
| Bloc SSM (recurrence style Mamba, memoire O(1)) | Disponible |
| Bloc hybride SSM/Attention 3:1 | Disponible |
| Rotation Hadamard (QuaRot) pour lissage d'activations | Disponible |
| Recuit de quantification (Quantization Annealing) | Disponible |
| Facteurs d'echelle alpha appris (QAT) | Disponible |
| Distillation KL + Feature Matching Loss | Disponible |
| Planificateur WSD (Warmup-Stable-Decay) | Disponible |
| Optimiseur decouple (WD=0 pour ternaire, WD=0,1 pour embedding) | Disponible |
| Pipeline d'entrainement Accelere | Disponible |
| Export SafeTensors compatible HuggingFace | Disponible (v0.2.0) |
| Rapport de compression automatique | Disponible (v0.2.0) |
| Generation avec repetition penalty + streaming | Disponible (v0.2.0) |
| Templates de chat (ChatML, Llama-3) | Disponible (v0.2.0) |
| Suite d'evaluation (perplexite, zero-shot, vitesse) | Disponible (v0.2.0) |
| Runtime C++ autonome (inference.h, zero dependance) | Disponible (v0.2.0) |
| **T-LoRA / BitDelta (adaptateurs ternaires low-rank)** | **Nouveau v0.3.0** |
| **OmniQuant (echelle S apprise activation-poids)** | **Nouveau v0.3.0** |
| **BitAttention (KV-Cache quantifie 2-bit)** | **Nouveau v0.3.0** |
| **Ternary MoE (Melange d'Experts ternaires)** | **Nouveau v0.3.0** |
| **WebGPU / WebAssembly backend navigateur** | **Nouveau v0.3.0** |
| Projection de taille 1 Gio (942 Mio, 4,07 milliards de parametres) | Disponible |
| **Pipeline modulaire (`TernairPipeline`)** | **Nouveau v0.5.0** |
| **Sauvegarde atomique de checkpoint (anti-corruption)** | **Nouveau v0.5.0** |
| **Estimateur memoire pre-flight (`estimate_memory`)** | **Nouveau v0.5.0** |
| **Profils intermediaires (`small`, `medium`, `large`)** | **Nouveau v0.5.0** |
| **Validation renforcee de `TernairConfig`** | **Nouveau v0.5.0** |

## Demarrage rapide

```bash
# Creer l'environnement
python3 -m venv .venv
source .venv/bin/activate

# Installer ternair depuis GitHub
pip install git+https://github.com/Simonc44/Ternair.git

# Afficher la configuration par defaut
python -m ternair info

# Projection de taille pour la cible 1 Gio
python -m ternair size --profile one_gb

# Executer un petit modele de demonstration
python -m ternair demo --profile tiny

# Exporter un modele glace en format SafeTensors
python -c "
from ternair.model.size_profiles import tiny_profile
from ternair.model.modeling import TernairForCausalLM
from ternair.model.export import export_to_safetensors, print_compression_report

model = TernairForCausalLM(tiny_profile(storage='fastpacked'))
model.freeze_storage()
model.eval()
print_compression_report(model)
export_to_safetensors(model, 'mon_modele.safetensors')
"

# Ajouter des adaptateurs T-LoRA pour specialisation
python -c "
from ternair.model.size_profiles import tiny_profile
from ternair.model.modeling import TernairForCausalLM
from ternair.quantization.bitdelta import add_lora_to_model

model = TernairForCausalLM(tiny_profile(storage='fastpacked'))
registry = add_lora_to_model(model, rank=8)
print(f'Adaptateurs : {registry.count_params():,} parametres entrainables')
"

# Evaluer la perplexite du modele
python -c "
from ternair.model.size_profiles import tiny_profile
from ternair.model.modeling import TernairForCausalLM
from ternair.benchmark.eval import compute_perplexity
from ternair.training.data import CharTokenizer, toy_corpus

model = TernairForCausalLM(tiny_profile(storage='fastpacked'))
tok = CharTokenizer(toy_corpus())
ppl = compute_perplexity(model, tok, dataset_name='wikitext',
                         subset='wikitext-2-raw-v1', split='test',
                         max_tokens=1000)
print(f'Perplexite: {ppl.perplexity:.2f}')
"
```

## Generation avancee

```python
from ternair.model.generation import generate, generate_stream, format_chat_prompt
import torch

# Generation avec repetition penalty
output = generate(model, prompt, max_new_tokens=64,
                  temperature=0.8, top_k=40, top_p=0.9,
                  repetition_penalty=1.1)

# Streaming token par token
for token_tensor in generate_stream(model, prompt, max_new_tokens=32):
    print(tokenizer.decode([token_tensor.item()]), end='', flush=True)

# Template de chat ChatML
prompt = format_chat_prompt([
    {"role": "user", "content": "Raconte une histoire"}
], format="chatml")
```

## T-LoRA / BitDelta -- Specialisation sans retoucher le modele de base

Ajoutez des adaptateurs ternaires low-rank pour specialiser votre modele sur une tache (code, medecine, langue) sans modifier les poids de base.

```python
from ternair.quantization.bitdelta import (
    add_lora_to_model, add_bitdelta_to_model,
    TernaryLoRALinear, AdapterRegistry
)

# LoRA ternaire (rank=8, ~95% de reduction)
registry = add_lora_to_model(model, rank=8, alpha=1.0)

# Entrainer seulement les adaptateurs
optimizer = torch.optim.AdamW(registry.adapter_params(), lr=1e-3)
for step in range(100):
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1)
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Sauvegarder les adaptateurs (quelques Ko)
torch.save(registry.state_dict(), "adaptateurs_code.pt")

# Recharger sur un autre modele de base
registry2 = add_lora_to_model(model2, rank=8)
registry2.load_state_dict(torch.load("adaptateurs_code.pt"))
```

## OmniQuant -- Calibration des echelles activation-poids

Optimise une echelle diagonale S pour minimiser l'erreur de quantification entre la sortie FP16 et la sortie ternaire + 8-bit activations. Reduit la perte de perplexite de 15 a 20%.

```python
from ternair.quantization.activation import ScaleEquivalence, calibrate_scale_equivalence

# Calibration automatique sur quelques echantillons
scales = calibrate_scale_equivalence(
    model,
    calibration_data=x_calib,
    lr=1e-3,
    steps=100,
)

# Application manuelle
scale = ScaleEquivalence(hidden_size=256)
x_s, w_s = scale(x, weight)  # (X * S^-1) x (S * W) = X x W
```

## BitAttention -- KV-Cache quantifie en 2-bit

Reduit l'empreinte memoire du KV-Cache par 4 ou 8, rendant les contextes ultra-longs (32k+ tokens) praticables sur appareils a faible RAM.

```python
from ternair.model.config import TernairConfig

# Activer la quantification KV avec kv_cache_bits=2
config = TernairConfig(
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    kv_cache_bits=2,  # 2-bit KV cache (BitAttention)
)
```

## Ternary MoE -- Melange d'Experts ternaires

Seulement 2 experts sur 8 sont actifs par token. Un modele de 12 milliards de parametres ternaires ne consomme le calcul que d'un modele de 2 milliards par token.

```python
from ternair.model.moe import TernaryMoEBlock, add_moe_to_model

# Remplacer certains MLP par des blocs MoE
add_moe_to_model(
    model,
    num_experts=8,    # 8 experts au total
    top_k=2,          # 2 actifs par token
    moe_layer_period=2,  # Toutes les 2 couches
)
```

## WebGPU / WebAssembly -- Inference dans le navigateur

Le backend WebGPU genere des compute shaders WGSL pour executer Ternair directement dans Chrome, Firefox ou Edge sans backend serveur.

```bash
# Generer les shaders WebGPU
python -c "
from ternair.kernels.webternair import (
    generate_wgsl_ternary_matmul,
    generate_js_runtime,
    validate_webgpu_kernels,
)

# Valider les shaders
results = validate_webgpu_kernels()
print(f'Shaders valides: {results}')

# Generer le runtime JS
js_code = generate_js_runtime()
with open('ternair-web.js', 'w') as f:
    f.write(js_code)
print('Runtime JS genere: ternair-web.js')
"
```

## Export HuggingFace

```python
from ternair.model.export import export_huggingface_package

# Export complet (config.json + model.safetensors + README.md)
export_huggingface_package(
    model,
    output_dir="./mon-modele-ternaire",
    model_name="MonModele-Ternair"
)
```

## Benchmark

```python
from ternair.benchmark.eval import run_eval_suite, print_report

report = run_eval_suite(model, tokenizer,
    run_perplexity=True, run_speed=True)
print_report(report)
```

## Runtime C++ autonome

Le fichier `src/ternair/kernels/inference.h` est un moteur d'inference header-only en C++17, zero dependance :

```cpp
#include "ternair/inference.h"

TernairRuntime runtime;
runtime.load("mon_modele.safetensors");

std::vector<int> tokens = {1, 234, 567, ...};
auto result = runtime.generate(tokens, 64);
```

Utilisable depuis n'importe quel langage via l'API C (`ternair_create`, `ternair_load`, `ternair_generate`).

## Entrainement

```bash
# Test de fumee (20 pas, modele tiny, 2 millions de parametres)
accelerate launch scripts/train.py --config scripts/train_tiny.yaml

# Entrainement complet a 1 Gio (60 couches, 2560 dimensions cachees, 4 milliards de parametres)
accelerate launch scripts/train.py --config scripts/train_one_gb.yaml

# Distillation QAT depuis un modele HuggingFace
PYTHONPATH=src python scripts/qat_distill.py
```

## Architecture

```
TernairConfig:
  ├── hidden_size=2560, num_hidden_layers=60
  ├── num_attention_heads=32, num_key_value_heads=8  (GQA)
  ├── intermediate_size=5120, max_position_embeddings=2048
  ├── storage: "packed" | "fastpacked" | "int8"
  ├── kv_cache_bits: 0 | 2 | 4  (BitAttention)
  ├── num_experts: 1 | 4 | 8 | 16  (Ternary MoE)
  │
  ├── ThalamicBottleneck (optionnel)
  │   └── K-WTA: top-32 tokens -> cross-attention -> 32 latents
  │
  ├── TernairHybridBlock x num_hidden_layers
  │   ├── Attention (GQA + RoPE + KV quant)  -- pattern periodique 3:1
  │   └── TernarySSM (scan style Mamba)       -- memoire O(1)
  │
  ├── TernaryMoEBlock (optionnel, remplace certains MLP)
  │   └── 8 experts ternaires, top-2 actifs
  │
  ├── AdapterRegistry (T-LoRA / BitDelta)
  │   └── Adaptateurs low-rank ternaires
  │
  └── TernairForCausalLM
      ├── TernairEmbedding (poids lies)
      ├── TernairModel (blocs hybrides + MoE)
      └── TernairLMHead (lie)
```

## Fonctionnement

### Quantification ternaire

Chaque matrice de poids `W` est quantifiee par ligne :

```
gamma = mean(|W|)                     # echelle par ligne
W_norm = W / gamma
W_t = round(clamp(W_norm, -1, 1))    # -> {-1, 0, +1}
```

**Propagation avant** : `y = (gamma * W_t) (x)` -> se reduit a des additions et soustractions d'activations.

**Retropropagation** : Straight-Through Estimator (STE) -- le gradient traverse `round` comme une identite.

### Recuit de quantification

Pendant le QAT, la temperature beta augmente progressivement de 1.0 a 15.0 :
```
W_proxy = tanh(beta * W / alpha) * alpha
```
Transition douce (tanh) -> dure (round) sans rupture de gradient.

### Factor d'echelle alpha appris

Chaque canal apprend son facteur d'echelle :
```
W_quant = round(clamp(W / alpha, -1, 1)) * alpha
```
Erreur de quantification reduite de ~40% par rapport a gamma statique.

### Conditionnement rapide (2 bits)

Chaque trit `{-1, 0, +1}` est stocke sur 2 bits dans un octet `uint8` (4 trits/octet) :

```python
# Decodage : sans modulo, sans table de correspondance, sans branchement
trit = (bits & 1) - ((bits >> 1) & 1)  # -> {-1, 0, +1}
```

### Rotation Hadamard (QuaRot)

Avant quantification INT8 des activations, une transformee de Hadamard rapide O(n log n) redistribue les outliers sur toutes les dimensions, reduisant l'erreur de quantification sans ajouter de parametres.

### OmniQuant -- Echelle equivalente S

Apprend une matrice diagonale S telle que `(X * S^-1) x (S * W) = X x W`. Optimisee pendant la calibration pour minimiser la distance de reconstruction entre le bloc FP16 et le bloc ternaire. Reduit la perte de perplexite de 15 a 20%.

### BitAttention -- KV-Cache quantifie

Les cles et valeurs du cache d'attention sont quantifiees en 2-bit par blocs de 32 tokens avec facteur d'echelle dynamique. L'empreinte memoire du KV-cache est divisee par 4 ou 8, rendant les fenetres de contexte 32k+ tokens praticables.

### T-LoRA / BitDelta

Decomposition des ajustements de poids en matrices de rang bas ternarisees `A x B` ou `A, B in {-1, 0, +1}`. Chaque adaptateur ne prend que quelques Ko pour specialiser le modele (code, medical, francais) sans toucher au coeur.

### Ternary MoE

Combinaison de la quantification 1.58-bit avec une architecture a Melange d'Experts. Seulement 2 experts sur 8 sont actifs par token. Le routage binaire selectionne les experts via un simple softmax + top-k.

### Projection memoire SSD

| Composant | Taille (profil 1 Gio) |
|-----------|-----------------------|
| Poids ternaires (conditionnes) | 776,2 Mio |
| Embedding + tete LM (lies) | 160,0 Mio |
| Echelles gamma | 5,1 Mio |
| RMSNorm + tampons divers | 0,6 Mio |
| **Total** | **941,9 Mio (< 1 Gio)** |

### Rapport de compression

```python
from ternair.model.export import print_compression_report
print_compression_report(model)

# ============================================================
#   Ternair Compression Report
# ============================================================
#   Total parameters      :    3,670,016
#   Ternary parameters    :    3,670,016
#   Storage mode          :   fastpacked
# -----------------------------------------------------------
#   FP16 equivalent       :       7.01 MiB
#   Ternair size          :       0.52 MiB
#   Compression ratio     :      13.56x
#   Savings               :      92.6%
# ============================================================
```

## Structure du projet

```
src/ternair/
├── quantization/
│   ├── linear.py         # TernairLinear avec alpha appris
│   ├── ternary.py        # Recuit beta, ternarization STE
│   ├── activation.py     # Hadamard + 8-bit + OmniQuant (v0.3.0)
│   ├── bitdelta.py       # T-LoRA / BitDelta adapters (NOUVEAU v0.3.0)
│   ├── distillation.py   # KL loss, Feature Matching, conversion HF
│   └── packing.py        # Conditionnement base-3 et 2-bit
├── kernels/
│   ├── inference.h       # Runtime C++ header-only
│   ├── webternair.py     # WebGPU / Wasm backend (NOUVEAU v0.3.0)
│   ├── cpu_matmul.h      # AVX-512 / ARM NEON matmul
│   └── ...
├── model/
│   ├── config.py         # Configuration + kv_cache_bits, num_experts
│   ├── attention.py      # GQA + KV quant 2-bit (BitAttention v0.3.0)
│   ├── moe.py            # Ternary MoE experts (NOUVEAU v0.3.0)
│   ├── export.py         # Export SafeTensors + HuggingFace
│   ├── generation.py     # Sampling avance, streaming, chat
│   └── ...
├── training/             # Planificateur WSD, optimiseur, entraineur
├── benchmark/
│   └── eval.py           # Suite d'evaluation
├── cli.py                # Point d'entree CLI
└── README.md             # Documentation complete

scripts/
├── train.py              # Point d'entree accelerate
├── train_tiny.yaml       # Configuration de test
├── train_one_gb.yaml     # Configuration 1 Gio
├── demo_reel.py          # Pipeline de demonstration complet
├── qat_distill.py        # Distillation QAT pour Colab
├── test_ci.py            # Tests CI (8 tests, v0.3.0)
└── wordy_colab.py        # Script Colab Wordy
```

## Performances

- 4,07 milliards de parametres ternaires stockes dans 942 Mio
- Multiplication matricielle par additions et soustractions uniquement -- zero multiplication FP en inference
- Compression 13.56x par rapport au FP16 equivalent
- Memoire O(1) pour la generation (mode SSM -- sans cache KV)
- Compression K-WTA : toute longueur d'entree -> 32 latents fixes
- KV-Cache 2-bit (BitAttention) : memoire divisee par 4 ou 8 pour contextes longs
- MoE : 12 milliards de parametres, cout de calcul d'un modele de 2 milliards

## Licence

Apache 2.0

Construit sur les travaux de recherche BitNet b1.58 de Microsoft Research (2024), BitDelta (2024), OmniQuant (2024), et BitMoE (2025).
