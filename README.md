# INF8225 — Projet Final — Équipe 12
## Analyse Architecturale de Whisper pour l'ASR en Français

**Hamza El Haddad · Jules Gouy**  
École Polytechnique de Montréal — Avril 2026

---

## 📋 Description

Étude architecturale systématique de [Whisper](https://github.com/openai/whisper) (OpenAI) appliquée à la reconnaissance automatique de la parole (ASR) en français sur le corpus **MLS French** (Multilingual LibriSpeech).

Trois axes d'analyse :

| Axe | Sujet |
|:---:|:------|
| 1 | **Scaling laws** — WER & latence vs. nombre de paramètres (small / medium / large-v3) |
| 2 | **Compromis vitesse/précision** — latence d'inférence vs. WER |
| 3 | **Fine-tuning avec données limitées** — ablations LR, gel encodeur, augmentation audio |

---

## 📁 Structure du dépôt

```
.
├── README.md
├── INF8225_Whisper_Analysis_v3.ipynb    # Notebook principal
├── rapport/
│   ├── rapport_INF8225.tex              # Rapport LaTeX (format IJCAI)
│   ├── ijcai19.sty                      # Style IJCAI
│   ├── fig_losses.png                   # Figure 1 : train/loss + eval/loss (WandB)
│   └── eval_wer_only.png               # Figure 2 : eval/WER (WandB)
└── slides/
    └── INF8225_Equipe12_Slides_v2.pptx  # Diapositives (12 slides)
```

---

## 🚀 Mise en route

### Prérequis

```bash
pip install transformers datasets accelerate evaluate jiwer
pip install wandb audiomentations
pip install torch torchaudio
```

### Exécution (Google Colab recommandé — GPU Tesla T4)

```python
# 1. Configurer WandB
import wandb
wandb.login()

# 2. Définir les variables globales en haut du notebook
WANDB_PROJECT = "inf8225-whisper-french"
WANDB_ENTITY  = "<votre-entity-wandb>"   # nom d'utilisateur ou équipe
```

Ensuite exécuter les cellules dans l'ordre. Chaque expérience vérifie si un artifact WandB existe déjà avant de relancer le calcul — **le notebook est entièrement reproductible à froid**.

> ⚠️ Les cellules de fine-tuning nécessitent un GPU. Sans GPU, seule l'évaluation zéro-shot (Axe 1) est exécutable.

---

## 📊 Résultats

### Axe 1 — Scaling Laws (zéro-shot, 500 ex. test)

| Variante | Paramètres | WER (%) | CER (%) | Latence (ms) |
|:---------|:----------:|:-------:|:-------:|:------------:|
| small | 244 M | 14,48 | 6,55 | 1 220 |
| medium | 769 M | 9,38 | 4,05 | 2 740 |
| large-v3 | 1 550 M | **6,17** | **3,99** | 3 063 |

### Axe 2 — Compromis vitesse / précision

| Modèle | WER (%) | Latence (ms) | Ratio WER/latence |
|:-------|:-------:|:------------:|:-----------------:|
| small | 14,48 | 1 220 | meilleur débit |
| medium | 9,38 | 2 740 | équilibré |
| large-v3 | 6,17 | 3 063 | meilleure précision |

→ Gain de WER de ×2,5 pour un coût de latence de ×2,5 entre `small` et `large-v3`.

### Axe 3 — Fine-tuning & Ablations (Whisper small, val 300 ex.)

| Configuration | WER val (%) | Δ pp vs. baseline |
|:--------------|:-----------:|:-----------------:|
| Baseline (zéro-shot) | 14,48 | — |
| FT, lr = 5×10⁻⁵ | 13,71 | −0,77 |
| FT, lr = 3×10⁻⁵ | 13,14 | −1,34 |
| FT, lr = 1×10⁻⁵ | 12,58 | −1,90 |
| FT, encodeur gelé | 13,50 | −0,98 |
| FT, encodeur libre (lr=1e-5) | 12,58 | −1,90 |
| **FT + augmentation audio** | **11,87** | **−2,61** |

**Test final** (500 ex.) : WER = **13,41 %**, CER = **6,12 %** (small, lr=1e-5)

---

## ⚙️ Configuration expérimentale

| Paramètre | Valeur |
|:----------|:-------|
| Modèles | `openai/whisper-small`, `openai/whisper-medium`, `openai/whisper-large-v3` |
| Dataset | `facebook/multilingual_librispeech` (fr) |
| Train / Val / Test | 2 000 / 300 / 500 exemples |
| GPU | Tesla T4 (15,6 Go VRAM) — Google Colab |
| Époques | 3 |
| Batch size | 16 |
| Optimizer | AdamW |
| LR scheduler | linear avec warmup |
| Warmup steps | 500 |
| Beam search | b = 5 |
| Suivi | Weights & Biases (`inf8225-whisper-french`) |

### Augmentation audio (`audiomentations`)

| Transformation | Paramètre |
|:---------------|:----------|
| `AddGaussianNoise` | σ ∈ [0.001, 0.01] |
| `TimeStretch` | ratio ∈ [0.9, 1.1] |
| `PitchShift` | ±2 demi-tons |

---

## 🗂️ Artifacts WandB

Tous les résultats sont persistés comme **artifacts Weights & Biases** sous le projet `inf8225-whisper-french`. Le notebook vérifie automatiquement leur existence avant de relancer un calcul.

| Artifact | Nom WandB | Type | Contenu |
|:---------|:----------|:----:|:--------|
| Évaluation zéro-shot | `inf8225-whisper-french-baseline` | `evaluation` | `baseline_results.json` — WER/CER par variante |
| Fine-tuning | `inf8225-whisper-french-finetuning` | `model` | Checkpoint `whisper-small` fine-tuné |
| Ablation LR | `inf8225-whisper-french-ablation-lr` | `ablation` | `ablation_lr.json` — WER val par LR |
| Ablation gel encodeur | `inf8225-whisper-french-ablation-freeze` | `ablation` | `ablation_freeze.json` — WER val enc. gelé |
| Ablation augmentation | `inf8225-whisper-french-ablation-augment` | `ablation` | `ablation_augment.json` — WER val + augmentation |
| Figures finales | `inf8225-whisper-french-figures` | `figures` | Toutes les figures exportées |

### Nommage des runs WandB

| Run name | Contenu |
|:---------|:--------|
| `baseline-evaluation` | Évaluation zéro-shot des 3 variantes |
| `fine-tuning-lr1e-05` | Fine-tuning small, lr=1e-5 (config retenue) |
| `ablation-lr` | Ablation taux d'apprentissage |
| `ablation-freeze` | Ablation gel de l'encodeur |
| `ablation-augmentation` | Fine-tuning + augmentation audio |
| `figures-final` | Export des figures WandB |
| `ckpt-<step>` | Checkpoints intermédiaires |

---

## 📈 Figures WandB

Les figures ci-dessous sont issues du suivi WandB en temps réel.

### Courbes d'entraînement

**train/loss & eval/loss** — Convergence des différentes configurations :

![train/loss et eval/loss](rapport/fig_losses.png)

Observations :
- `fine-tuning-lr1e-05` (rouge/jaune) — convergence la plus basse en validation → LR optimal
- `ablation-lr` (LR=3e-5, orange) — perte initiale plus haute : mise à jour trop agressive
- `ablation-freeze` (bleu) — perte val plus élevée : geler l'encodeur limite l'adaptation
- Convergence rapide dès **100 pas** pour toutes les configs → pré-entraînement très efficace

**eval/WER** — WER de validation au fil de l'entraînement :

![eval/WER](rapport/eval_wer_only.png)

### Autres métriques WandB (runs disponibles)

| Métrique | Description |
|:---------|:------------|
| `train/loss` | Perte d'entraînement par pas |
| `train/learning_rate` | Évolution du LR (scheduler linéaire + warmup) |
| `train/grad_norm` | Norme du gradient — détection d'instabilité |
| `eval/loss` | Perte de validation après chaque époque |
| `eval/wer` | WER de validation |
| `eval/runtime` | Temps d'évaluation (secondes) |
| `eval/samples_per_second` | Débit d'évaluation |
| `test/wer` | WER final sur le jeu de test (500 ex.) |
| `test/cer` | CER final sur le jeu de test (500 ex.) |
| `ablation/lr/1e-05/wer` | WER val pour lr=1e-5 |
| `ablation/lr/3e-05/wer` | WER val pour lr=3e-5 |
| `ablation/lr/5e-05/wer` | WER val pour lr=5e-5 |

---

## 📦 Modèles Hugging Face

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Zéro-shot
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Forcer le français
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="french", task="transcribe"
)
```

Variantes disponibles : `openai/whisper-small` · `openai/whisper-medium` · `openai/whisper-large-v3`

---

## 📄 Rapport

Le rapport (format IJCAI, 4 pages) est dans le dossier `rapport/`.

**Compiler localement :**
```bash
cd rapport/
pdflatex rapport_INF8225.tex
pdflatex rapport_INF8225.tex   # second passage pour les références croisées
```

**Éditer sur Overleaf :** importer le fichier `INF8225_Overleaf.zip` via *New Project → Upload Project*.  
Le zip contient : `rapport_INF8225.tex` · `ijcai19.sty` · `fig_losses.png` · `eval_wer_only.png`.

---

## 🔗 Liens

| Ressource | Lien |
|:----------|:-----|
| Paper Whisper | [arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356) |
| MLS Dataset | [HuggingFace — multilingual_librispeech](https://huggingface.co/datasets/facebook/multilingual_librispeech) |
| Whisper small | [HuggingFace — openai/whisper-small](https://huggingface.co/openai/whisper-small) |
| WandB project | [wandb.ai — inf8225-whisper-french](https://wandb.ai) |
| Seq2SeqTrainer | [HuggingFace Transformers docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer) |

---

## 👥 Équipe 12

| Nom | Email |
|:----|:------|
| Hamza El Haddad | hamza.el-haddad@polymtl.ca |
| Jules Gouy | jules.gouy@polymtl.ca |

*INF8225 — Techniques probabilistes et d'apprentissage en IA*  
*École Polytechnique de Montréal — Avril 2026*
