# Politique de Securite Ternair

Ce document explique comment signaler une vulnerabilite de securite dans
Ternair et ce que vous pouvez attendre de l'equipe de maintenance.

---

## 1. Versions prises en charge

| Branche | Statut | Support |
|---------|--------|---------|
| `main` | active | Correctifs reguliers, securite incluse |
| Branche `fix/*` critique | courte duree | Correctifs de securite uniquement |
| Anciennes releases (`v0.5.x`, `v0.4.x`, ...) | obsoletes | Aucune mise a jour, migrer vers `main` |

Seule la branche `main` recoit des mises a jour de securite
backportees pour les versions `vX.Y.Z` recentes (politique de
support detaillee dans `CONTRIBUTING.md` section 6).

## 2. Comment signaler une vulnerabilite

### A privilegier : GitHub Security Advisories

1. Allez sur
   [https://github.com/Simonc44/Ternair/security/advisories/new](https://github.com/Simonc44/Ternair/security/advisories/new).
2. Remplissez le formulaire avec un titre descriptif :
   `[vuln] courte description (composant affecte, version)`.
3. Joignez une description detaillee du probleme, du scenario
   d'exploitation et de l'impact potentiel.

### Alternative : email direct

Si vous ne pouvez pas utiliser GitHub, envoyez un email chiffre PGP
au mainteneur principal (voir le profil GitHub pour la cle).

## 3. Que inclure dans votre rapport

Pour accelerer le triage, merci de fournir :

- **Description** : nature de la vulnerabilite (XSS, RCE, DoS, fuite
  memoire, depassement de tenseur, etc.).
- **Reproduction** : script Python minimal, commande CLI, ou snippet
  de notebook Colab.
- **Impact** : que peut faire un attaquant (lecture de poids, plantage
  garanti, execution de code, biais de modele, etc.).
- **Version concernee** : commit SHA ou tag `vX.Y.Z`.
- **Environnement** : OS, version Python/Torch/triton, GPU/CPU,
  flag bitnet ou autre profil active.
- **Pyritisation** : trace complete si applicable (attention a ne pas
  inclure de poids ou de secrets).

## 4. Ce que vous pouvez attendre de nous

| Etape | Delai |
|-------|-------|
| Accuse de reception | sous 3 jours ouvrables |
| Triage preliminaire (severite CVSS approximee) | sous 7 jours ouvrables |
| Patch publie sur une branche privee | sous 30 jours (selon severite) |
| Disclosure publique apres coordination | variable, selon gravite |

Severite utilisee pour le triage :

| Niveau | Critere |
|--------|---------|
| Critique | RCE, corruption memoire, bypass complet du sandbox |
| Haute | Fuite de donnees, escalade de privileges, DoS non recuperable |
| Moyenne | DoS local, fuite d'info partielle, bias reproducible |
| Basse | Defauts mineurs, fuite d'info theorique requerrant conditions particulieres |

## 5. Politique de non-represailles

Ternair s'engage a ne prendre aucune action legale ou repressive
contre les chercheurs qui :
- Agissent de bonne foi et suivent ce processus de divulgation.
- Respectent la confidentialite jusqu'a la publication du patch.
- Ne depassent pas le scope strictement necessaire a la preuve de
  concept (pas d'exfiltration de donnees, pas d'attaque sur les
  autres utilisateurs).

Cette politique s'aligne sur les principes du
[disclosure coordinne (CERT/CC)](https://www.cert.org/vulnerability-analysis/research-work/security-coordination-disclosure.cfm).

## 6. En dehors du code source

Ternair etant un projet de recherche en IA locale (inference sur CPU
ou GPU de l'utilisateur), la surface d'attaque classique "server-side"
ne s'applique pas. Les principales categories concernees sont :

- **Biais / prompt injection** : si un modele charge des poids
  deliberablement corrompus et qu'une instruction utilisateur peut
  declencher un comportement dangereux.
- **Deni de service local** : OOM, segfault sur GPU, boucle infinie
  en generation (contexte non borne).
- **Fuite de poids** : si vous distribuez accidentellement un
  checkpoint contenant des donnees personnelles non anonymisees.
- **Supply chain** : si une dependance (torch, triton, bitsandbytes)
  introduit une regression via Ternair.

Pour les vulnerabilites dans les dependances amont, suivre les
politiques de leurs mainteneurs respectifs et considerer ouvrir une
issue Ternair pour tracker l'impact.

## 7. Hall of Fame

Les contributeurs ayant aide a ameliorer la securite du projet
seront mentionnes ici (avec leur permission) apres chaque divulgation
coordonnee.

_(Pas encore de contribution securite documentee.)_

---

Merci de contribuer a la securite de Ternair.
