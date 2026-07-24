[![CI](https://github.com/Simonc44/Ternair/actions/workflows/ci.yml/badge.svg)](https://github.com/Simonc44/Ternair/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.4%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/Simonc44/Ternair)](https://github.com/Simonc44/Ternair/releases)

# Ternair

**BitNet b1.58 -- Reseaux de neurones ternaires a l'echelle 1 Gio**

---

Ternair est une implementation de production de **BitNet b1.58**, une architecture de reseau de neurones dans laquelle chaque poids est contraint a `{-1, 0, +1}` (valeurs ternaires, ~1,58 bits). Cette approche permet :

- Compression memoire ~16x par rapport au FP16 (942 Mio pour un modele de 4 milliards de parametres)
- Arithmetique par additions et soustractions uniquement -- zero multiplication flottante durant l'inference
- Absence de cache KV lors de l'utilisation des couches SSM optionnelles (memoire de generation en O(1))
- Compression de tokens K-WTA via le goulot thalamique (ThalamicBottleneck) : 32 latents fixes par sequence

## Fonctionnalites

| Fonctionnalite | Statut |
|----------------|--------|
| Quantification ternaire (STE, echelle gamma, 3 valeurs) | Disponible |
| Conditionnement : base-3 (1,6 b/v) ou 2 bits (2,0 b/v) | Disponible |
| Noyau GPU (Triton -- decodage en boucle JIT) | Disponible |
| Noyau CPU (C++ SIMD -- AVX-512 / ARM NEON) | Disponible |
| Modele causal LM (attention GQA, RoPE, MLP SquaredReLU) | Disponible |
| ThalamicBottleneck (compression K-WTA, K=32) | Disponible |
| Bloc SSM (recurrence style Mamba, memoire O(1)) | Disponible |
| Planificateur WSD (Warmup-Stable-Decay) | Disponible |
| Optimiseur decouple (WD=0 pour ternaire, WD=0,1 pour embedding) | Disponible |
| Pipeline d'entrainement Accelere | Disponible |
| Projection de taille 1 Gio (942 Mio, 4,07 milliards de parametres) | Disponible |

## Demarrage rapide

```bash
# Creer l'environnement
python3 -m venv .venv
source .venv/bin/activate

# Cloner et installer ternair depuis GitHub
git clone https://github.com/Simonc44/Ternair.git
cd Ternair
pip install -e .

# Afficher la configuration par defaut
python -m ternair info

# Projection de taille pour la cible 1 Gio
python -m ternair size --profile one_gb

# Executer un petit modele de demonstration
python -m ternair demo --profile tiny

# Lancer tous les tests
python -m pytest tests/ --confcutdir=tests
```

## Entrainement

```bash
# Test de fumee (20 pas, modele tiny, 2 millions de parametres)
accelerate launch scripts/train.py --config scripts/train_tiny.yaml

# Entrainement complet a 1 Gio (60 couches, 2560 dimensions cachees, 4 milliards de parametres)
accelerate launch scripts/train.py --config scripts/train_one_gb.yaml
```

## Architecture

```
TernairConfig:
  ├── hidden_size=2560, num_hidden_layers=60
  ├── num_attention_heads=32, num_key_value_heads=8  (GQA)
  ├── intermediate_size=5120, max_position_embeddings=2048
  ├── storage: "packed" | "fastpacked" | "int8"
  │
  ├── ThalamicBottleneck (optionnel)
  │   └── K-WTA: top-32 tokens -> cross-attention -> 32 latents
  │
  ├── TernairHybridBlock x num_hidden_layers
  │   ├── Attention (GQA + RoPE)          -- premieres couches `num_attn_layers`
  │   └── TernarySSM (scan style Mamba)   -- couches restantes
  │
  └── TernairForCausalLM
      ├── TernairEmbedding (poids lies)
      ├── TernairModel (blocs hybrides)
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

### Conditionnement rapide (2 bits)

Chaque trit `{-1, 0, +1}` est stocke sur 2 bits dans un octet `uint8` (4 trits/octet) :

```python
# Decodage : sans modulo, sans table de correspondance, sans branchement
trit = (bits & 1) - ((bits >> 1) & 1)  # -> {-1, 0, +1}
```

### Projection memoire SSD

| Composant | Taille (profil 1 Gio) |
|-----------|-----------------------|
| Poids ternaires (conditionnes) | 776,2 Mio |
| Embedding + tete LM (lies) | 160,0 Mio |
| Echelles gamma | 5,1 Mio |
| RMSNorm + tampons divers | 0,6 Mio |
| **Total** | **941,9 Mio (< 1 Gio)** |

## Structure du projet

```
src/ternair/
├── quantization/     # STE, conditionnement, TernairLinear
├── kernels/          # Triton GPU, C++ SIMD, reference numpy
├── model/            # Configuration, attention, MLP, SSM, thalamus, generation
├── training/         # Planificateur WSD, optimiseur, entraineur
├── benchmark/        # Projection de taille
├── cli.py            # Point d'entree CLI
└── README.md         # Documentation complete

scripts/
├── train.py          # Point d'entree accelerate
├── train_tiny.yaml   # Configuration de test
└── train_one_gb.yaml # Configuration 1 Gio
```

## Performances

- 4,07 milliards de parametres ternaires stockes dans 942 Mio
- Multiplication matricielle par additions et soustractions uniquement -- zero multiplication FP en inference
- Memoire O(1) pour la generation (mode SSM -- sans cache KV)
- Compression K-WTA : toute longueur d'entree -> 32 latents fixes

## Licence

Apache 2.0

Construit sur les travaux de recherche BitNet b1.58 de Microsoft Research (2024).
