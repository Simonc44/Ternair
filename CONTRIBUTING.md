# Contribuer a Ternair

Merci de votre interet pour Ternair. Ce guide couvre tout ce dont vous
avez besoin pour proposer un patch, un bug-fix ou une nouvelle
fonctionnalite.

---

## 1. Mise en place de l'environnement

```bash
git clone https://github.com/Simonc44/Ternair.git
cd Ternair

python3 -m venv .venv
source .venv/bin/activate   # PowerShell : .venv\Scripts\Activate.ps1

pip install -U pip setuptools wheel
pip install -e .
```

Dependances minimales (deja declarees dans `pyproject.toml`) :
- Python >= 3.10
- torch >= 2.4
- numpy >= 1.17

Dependances de dev (optionnel) :
- `pytest` pour des runs de tests interactifs
- `triton` pour executer les kernels GPU en local
- `bitsandbytes` pour les notebooks Colab de distillation

---

## 2. Workflow de developpement

```bash
# Creer une branche
git checkout -b feat/ma-fonctionnalite

# Editer le code sous src/ternair/ ou scripts/
# Ajouter un test dans scripts/test_ci.py si pertinent

# Verifier que tout passe avant de commit
PYTHONPATH=src python3 scripts/test_ci.py
```

### Ou placer le code

| Domaine | Emplacement |
|---------|-------------|
| Quantification, packing, distillation | `src/ternair/quantization/` |
| Kernels bas niveau (Triton, C++, WebGPU) | `src/ternair/kernels/` |
| Modele, generation, export | `src/ternair/model/` |
| Pipeline d'entrainement | `src/ternair/training/` |
| Benchmarks de taille et qualite | `src/ternair/benchmark/` |
| Scripts utilisateurs (train, distill, demo) | `scripts/` |

### Conventions de code

- Python : pep-8, annotations de type `from __future__ import annotations`.
- Pas d'emoji dans le code, les commentaires ou la documentation.
- Les docstrings publiques en anglais (compatibles Sphinx plus tard).
- Les messages utilisateur (README, FEATURES, CLI) en francais.
- Eviter les dependances externes au-dela de `torch` + `numpy` pour le
  coeur du package.

---

## 3. Tests

Le projet utilise un runner CI minimaliste (`scripts/test_ci.py`) qui
n'a pas besoin de pytest ou d'un GPU. C'est le seul test a executer
avant de proposer une PR :

```bash
PYTHONPATH=src python3 scripts/test_ci.py
```

Tous les tests doivent etre verts (17 tests passes en v0.6.0). Si vous
ajoutez une fonctionnalite, ajoutez au moins un test dedie (avec un
nom explicite `test_xxx`) et stitch au runner via `main()`.

---

## 4. Messages de commit

Nous suivons un format proche de Conventional Commits, en francais ou
en anglais, avec un prefix court :

| Prefix | Usage |
|--------|-------|
| `feat:` | Nouvelle fonctionnalite |
| `fix:` | Correction de bug |
| `refactor:` | Restructuration sans changement de comportement |
| `perf:` | Optimisation de performance ou d'empreinte memoire |
| `docs:` | Documentation seule (README, FEATURES, MD) |
| `test:` | Ajout ou correction de tests uniquement |
| `chore:` | Taches de maintenance (CI, deps, format) |

Exemples :
```
feat(packing): ajout du codec base8 canonique
fix(ssm): overflow exp(-log_A_cumsum) sur sequences >= 12 tokens
refactor(kernels): unifier triton_fast single + batched
```

---

## 5. Branches et Pull Requests

- Branche principale : `main`.
- Branches de feature : `feat/<court-desc>`, `fix/<court-desc>`,
  `refactor/<court-desc>` ou `docs/<court-desc>`.
- Une PR = un seul sujet, preferablement petite (< 400 lignes diff).
- Decrivez le **pourquoi** avant le **comment** dans la description.
- Listez les tests executes et le resultat.
- Si la PR touche la numerique (loss, perplexite, overflow), joignez
  un court tableau avant/apres.

---

## 6. Numerotation des versions

Ternair suit SemVer adapte :

- `MAJOR` (vX.0.0) : changement incompatible de l'API publique
  (export, signatures, chemins d'import canoniques).
- `MINOR` (v0.X.0) : ajout de fonctionnalite retro-compatible
  (kernels, profils, entrainement, generation).
- `PATCH` (v0.0.X) : correction de bug ou de perf strictement
  locale.

Le tag Git suit la forme `vM.m.p` et est pousse en meme temps que le
commit de la release par le mainteneur.

---

## 7. Signaler un bug ou demander une fonctionnalite

Ouvrez une **issue** sur GitHub avec :
- Un titre clair : `[bug]` ou `[enhancement]` + phrase courte.
- La version concernee (`v0.X.Y` ou commit SHA).
- Le strict minimum de reproduction (script de 10 lignes idealement).
- Pour les bugs numeriques : le dtype, la sequence length, le profil
  utilise.
- Pour les bugs de perf : GPU/CPU, taille du modele, tokens/s observes.

---

## 8. Style editorial

- Francais sans emoji pour la documentation utilisateur.
- Anglais sans emoji pour les docstrings et commentaires de code.
- Tableaux plutot que des paragraphes pour les listes de plus de 3
  items.
- Pas de section "Introduction" sauf si elle apporte une information
  qu'on ne trouve pas dans le titre.
- Lier les fichiers par leur chemin relatif (`src/ternair/...`).

---

## 9. Licence

En contribuant, vous acceptez que vos contributions soient distribuees
sous la licence Apache 2.0 (voir `LICENSE`).

Merci de votre aide pour faire avancer Ternair.
