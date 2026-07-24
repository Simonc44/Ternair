# Ternair -- Fonctionnalites completes (v0.5.0)

Documentation exhaustive de toutes les fonctionnalites de Ternair,
classees par domaine.

## 0. Pipeline robuste pour modeles intermediaires (v0.5.0)

| Fonctionnalite | Fichier | Statut |
|---------------|---------|--------|
| `TernairPipeline` (build/train/distill/freeze/export) | `training/pipeline.py` | Nouveau v0.5.0 |
| `PipelineStage` (FSM du cycle de vie) | `training/pipeline.py` | Nouveau v0.5.0 |
| `AtomicCheckpointSaver` (write tmp -> os.replace) | `training/atomic.py` | Nouveau v0.5.0 |
| `MemoryEstimate` (pre-flight memoire) | `training/memory.py` | Nouveau v0.5.0 |
| `estimate_memory(model, batch, seq)` | `training/memory.py` | Nouveau v0.5.0 |
| Profils intermediaires `small`/`medium`/`large` | `model/size_profiles.py` | Nouveau v0.5.0 |
| `fit_profile_for_budget(target_mib)` | `model/size_profiles.py` | Nouveau v0.5.0 |
| Validation renforcee dans `TernairConfig.__post_init__` | `model/config.py` | Nouveau v0.5.0 |
| Fix bug precedence `optimizer.py` | `training/optimizer.py` | Nouveau v0.5.0 |

---

## 1. Quantification ternaire (BitNet b1.58)

| Fonctionnalite | Fichier | Statut |
|---------------|---------|--------|
| Poids ternaires `{-1, 0, +1}` avec STE | `quantization/ternary.py` | Stable |
| Facteur d'echelle gamma par ligne `mean(\|W\|)` | `quantization/ternary.py` | Stable |
| Alpha appris par canal (entrainable) | `quantization/linear.py` | Stable |
| Recuit de quantification (beta: 1.0 -> 15.0) | `quantization/ternary.py` | Stable |
| Fonction autograd personnalisee `TernaryLinearFn` | `quantization/ternary.py` | Stable |

### Quantification des activations

| Fonctionnalite | Fichier | Statut |
|---------------|---------|--------|
| 8-bit absmax per-token activation quant | `quantization/activation.py` | Stable |
| STE pour la retropropagation des activations | `quantization/activation.py` | Stable |
| Rotation Hadamard (QuaRot/SpinQuant) FWHT O(n log n) | `quantization/activation.py` | Stable |
| OmniQuant : echelle equivalente S apprise | `quantization/activation.py` | Nouveau v0.3.0 |

### Conditionnement (packing)

| Fonctionnalite | Fichier | Bits/valeur |
|---------------|---------|-------------|
| `int8` : 1 byte par trit | `quantization/packing.py` | 8.0 |
| `packed` : 5 trits/byte (base-3) | `quantization/packing.py` | 1.6 |
| `fastpacked` : 4 trits/byte (2-bit) | `kernels/packing_fast.py` | 2.0 |

### Stockage et compression

| Fonctionnalite | Fichier | Description |
|---------------|---------|-------------|
| Gel du stockage (`freeze_storage()`) | `quantization/linear.py` | Passe du FP a packed |
| Dequantisation pour inference | `quantization/linear.py` | Reconstruction gamma * trits |
| Rapport de compression automatique | `model/export.py` | Rapport FP16 vs ternaire |
| Projection memoire 1 Gio | `benchmark/size.py` | 4.07B params dans 942 Mio |

---

## 2. Architecture du modele

| Fonctionnalite | Fichier | Statut |
|---------------|---------|--------|
| Attention GQA avec projections ternaires | `model/attention.py` | Stable |
| RoPE (Rotary Position Embedding) | `model/attention.py` | Stable |
| KV-Cache quantifie 2-bit (BitAttention) | `model/attention.py` | Nouveau v0.3.0 |
| MLP SwiGLU ternaire (silu * gate) | `model/mlp.py` | Stable |
| Bloc SSM (Mamba-style, memoire O(1)) | `model/ssm.py` | Stable |
| Bloc hybride SSM/Attention 3:1 | `model/hybrid_block.py` | Stable |
| ThalamicBottleneck (K-WTA, 32 latents) | `model/thalamus.py` | Stable |
| RMSNorm (LayerNorm sans centrage) | `model/block.py` | Stable |
| TernairForCausalLM (tete LM liee) | `model/modeling.py` | Stable |
| Ternary MoE (8 experts, top-2 actifs) | `model/moe.py` | Nouveau v0.3.0 |

### Configuration (`TernairConfig`)

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| `vocab_size` | 32000 | Taille du vocabulaire |
| `hidden_size` | 2560 | Dimension cachee |
| `intermediate_size` | 6912 | Dimension MLP |
| `num_hidden_layers` | 24 | Nombre de couches |
| `num_attention_heads` | 32 | Tetes d'attention |
| `num_key_value_heads` | 4 | GQA KV heads |
| `attn_layer_period` | 4 | Pattern SSM/Attention 3:1 |
| `kv_cache_bits` | 0 | KV cache quantifie (2=actif) |
| `num_experts` | 1 | MoE desactive si =1 |
| `top_k_experts` | 1 | Top-K experts actifs |
| `storage` | "packed" | Mode de stockage |

### Profils pre-configures

| Profil | Hidden | Couches | Tetes | Parametres | Taille |
|--------|--------|---------|-------|-----------|--------|
| `tiny_profile()` | 256 | 8 | 4 | 2,6 M | ~2,5 Mio |
| `base_profile()` | 2560 | 24 | 32 | 700 M | ~150 Mio |
| `one_gb_profile()` | 2560 | 60 | 32 | 4,07 B | ~942 Mio |

---

## 3. Generation et pipeline chat

| Fonctionnalite | Fichier | Statut |
|---------------|---------|--------|
| Decodage greedy (argmax) | `model/generation.py` | Stable |
| Temperature sampling | `model/generation.py` | Stable |
| Top-K filtering | `model/generation.py` | Stable |
| Top-P nucleus sampling | `model/generation.py` | Stable |
| Repetition penalty | `model/generation.py` | Nouveau v0.2.0 |
| Streaming token par token | `model/generation.py` | Nouveau v0.2.0 |
| Chat template ChatML | `model/generation.py` | Nouveau v0.2.0 |
| Chat template Llama-3 | `model/generation.py` | Nouveau v0.2.0 |

---

## 4. Distillation et QAT

| Fonctionnalite | Fichier | Statut |
|---------------|---------|--------|
| KL divergence loss avec temperature | `quantization/distillation.py` | Stable |
| Perte combinee CE + KL | `quantization/distillation.py` | Stable |
| Conversion HuggingFace -> TernairLinear | `quantization/distillation.py` | Stable |
| Alpha appris par canal | `quantization/distillation.py` | Stable |
| Mixed-precision selective (embed FP16) | `quantization/distillation.py` | Stable |
| Feature Matching Loss | `quantization/distillation.py` | Nouveau v0.2.0 |
| HiddenStateCapture hooks | `quantization/distillation.py` | Nouveau v0.2.0 |

---

## 5. T-LoRA / BitDelta (adaptateurs ternaires)

| Fonctionnalite | Fichier | Statut |
|---------------|---------|--------|
| TernaryLoRALinear (low-rank A x B ternaire) | `quantization/bitdelta.py` | Nouveau v0.3.0 |
| TernaryDeltaLinear (full-rank ternaire) | `quantization/bitdelta.py` | Nouveau v0.3.0 |
| AdapterRegistry (gestion des adaptateurs) | `quantization/bitdelta.py` | Nouveau v0.3.0 |
| `add_lora_to_model()` | `quantization/bitdelta.py` | Nouveau v0.3.0 |
| `add_bitdelta_to_model()` | `quantization/bitdelta.py` | Nouveau v0.3.0 |

---

## 6. Export et formats

| Fonctionnalite | Fichier | Statut |
|---------------|---------|--------|
| SafeTensors (spec-compliant, sans dep) | `model/export.py` | Nouveau v0.2.0 |
| HuggingFace package (config.json + .safetensors) | `model/export.py` | Nouveau v0.2.0 |
| Rapport de compression automatique | `model/export.py` | Nouveau v0.2.0 |
| GGUF export (llama.cpp / Ollama compatible) | `kernels/gguf_export.py` | **Nouveau v0.4.0** |

---

## 7. Backends d'inference

| Backend | Fichier | Description | Statut |
|---------|---------|-------------|--------|
| PyTorch natif (autograd) | `quantization/linear.py` | Entrainement | Stable |
| NumPy reference | `kernels/packed_ops.py` | Validation | Stable |
| NumPy batched | `kernels/packed_ops.py` | Sequences multiple | Stable |
| Triton GPU (JIT compile) | `kernels/triton_matmul.py` | GPU matmul | Stable |
| **Triton batched GPU** | `kernels/triton_matmul_fused.py` | **GPU batch + attention** | **Nouveau v0.4.0** |
| C++ SIMD AVX-512 | `kernels/cpu_matmul.h` | CPU x86 | Stable |
| C++ SIMD ARM NEON | `kernels/cpu_matmul.h` | CPU ARM | Stable |
| **C++ header-only runtime** | `kernels/inference.h` | **Zero dep** | **Nouveau v0.2.0** |
| **WebGPU / Wasm** | `kernels/webternair.py` | **Browser** | **Nouveau v0.3.0** |

---

## 8. Evaluation et benchmarks

| Fonctionnalite | Fichier | Statut |
|---------------|---------|--------|
| Perplexite WikiText-2 (fenetre glissante) | `benchmark/eval.py` | Nouveau v0.2.0 |
| Perplexite C4 | `benchmark/eval.py` | Nouveau v0.2.0 |
| Zero-shot HellaSwag | `benchmark/eval.py` | Nouveau v0.2.0 |
| Zero-shot ARC-Challenge | `benchmark/eval.py` | Nouveau v0.2.0 |
| Zero-shot MMLU | `benchmark/eval.py` | Nouveau v0.2.0 |
| Benchmark vitesse (tokens/sec) | `benchmark/eval.py` | Nouveau v0.2.0 |
| Rapport structure EvalReport | `benchmark/eval.py` | Nouveau v0.2.0 |
| **Triton vs numpy benchmark** | `kernels/triton_matmul_fused.py` | **Nouveau v0.4.0** |
| Projection de taille | `benchmark/size.py` | Stable |

---

## 9. Entrainement

| Fonctionnalite | Fichier | Statut |
|---------------|---------|--------|
| Optimiseur decouple (WD=0 ternaire) | `training/optimizer.py` | Stable |
| Planificateur WSD | `training/scheduler.py` | Stable |
| Entraineur accelerate | `training/trainer.py` | Stable |
| DataLoader HuggingFace | `training/data.py` | Stable |
| Configuration YAML | `training/config.py` | Stable |

---

## 10. CLI (Commandes disponibles)

```bash
python -m ternair info --profile tiny           # Config du modele
python -m ternair size --profile one_gb          # Projection taille
python -m ternair demo --profile tiny            # Demo complete
python -m ternair train-one --profile tiny       # Test entrainement
```

---

## 11. Scripts fournis

| Script | Description |
|--------|-------------|
| `scripts/demo_reel.py` | Pipeline demo complet (6 etapes) |
| `scripts/train.py` | Entrainement avec accelerate |
| `scripts/qat_distill.py` | Distillation QAT HuggingFace |
| `scripts/colab_distill.py` | Distillation pour Google Colab |
| `scripts/wordy_colab.py` | Creer Wordy sur Colab |
| `scripts/test_ci.py` | Tests CI (8 tests v0.3.0, 10+ tests v0.4.0) |

---

## 12. Map des fichiers

```
src/ternair/
├── quantization/
│   ├── linear.py          # TernairLinear avec alpha appris
│   ├── ternary.py         # STE, ternarisation, recuit beta
│   ├── activation.py      # 8-bit activations + Hadamard + OmniQuant
│   ├── bitdelta.py        # T-LoRA / BitDelta adapters
│   ├── distillation.py    # KL loss, conversion HF, Feature Matching
│   ├── packing.py         # Conditionnement base-3
│   └── __init__.py
├── kernels/
│   ├── inference.h        # Runtime C++ header-only
│   ├── cpu_matmul.h       # C++ SIMD AVX-512 / NEON
│   ├── cpu_matmul.py      # Wrapper Python C++
│   ├── triton_matmul.py   # Kernel Triton GPU
│   ├── triton_matmul_fused.py  # Triton batched (NOUVEAU v0.4.0)
│   ├── gguf_export.py     # Export GGUF (NOUVEAU v0.4.0)
│   ├── webternair.py      # WebGPU / Wasm
│   ├── packing_fast.py    # 2-bit packing
│   ├── packed_ops.py      # Reference numpy
│   └── __init__.py
├── model/
│   ├── config.py          # TernairConfig
│   ├── modeling.py        # TernairForCausalLM
│   ├── attention.py       # GQA + BitAttention KV
│   ├── mlp.py             # SwiGLU ternaire
│   ├── block.py           # Decoder block + RMSNorm
│   ├── hybrid_block.py    # SSM/Attention 3:1
│   ├── ssm.py             # Selective scan Mamba
│   ├── thalamus.py        # K-WTA bottleneck
│   ├── moe.py             # Ternary MoE
│   ├── export.py          # SafeTensors + HuggingFace
│   ├── generation.py      # Sampling, streaming, chat
│   ├── size_profiles.py   # tiny/base/one_gb
│   └── __init__.py
├── training/
│   ├── data.py, config.py, optimizer.py, scheduler.py, trainer.py
├── benchmark/
│   ├── size.py            # Projection memoire
│   ├── eval.py            # Benchmark (perplexite, zero-shot)
│   └── __init__.py
├── cli.py                 # Ligne de commande
├── _version.py            # Version
└── __init__.py
```
