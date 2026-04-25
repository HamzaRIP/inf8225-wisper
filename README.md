# INF8225 — Projet Final — Équipe 12
## Analyse Architecturale de Whisper pour l'ASR en Français

**Hamza El Haddad · Jules Gouy**  
École Polytechnique de Montréal — Avril 2026

---

## 📋 Description

Étude architecturale systématique de [Whisper](https://github.com/openai/whisper) (OpenAI) appliquée à la reconnaissance automatique de la parole (ASR) en français sur le corpus **MLS French** (Multilingual LibriSpeech).

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
WANDB_ENTITY  = "<votre-entity-wandb>"
```

Chaque expérience vérifie si un artifact WandB existe déjà avant de relancer le calcul — **le notebook est entièrement reproductible à froid**.

> ⚠️ Les cellules de fine-tuning nécessitent un GPU. Sans GPU, seule l'évaluation zéro-shot (Axe 1) est exécutable.

---

## 📊 Résultats

### Axe 1 — Scaling Laws (zéro-shot, 500 ex. test)

| Variante | Paramètres | WER (%) | CER (%) | Latence (ms) |
|:---------|:----------:|:-------:|:-------:|:------------:|
| small | 244 M | 14,48 | 6,55 | 1 220 |
| medium | 769 M | 9,38 | 4,05 | 2 740 |
| large-v3 | 1 550 M | **6,17** | **3,99** | 3 063 |

→ WER divisé par **2,35** entre small et large-v3 pour une latence multipliée par **2,5** seulement.

### Axe 3 — Fine-tuning & Ablations (Whisper small, val 300 ex.)

| Configuration | WER val (%) | Δ pp |
|:--------------|:-----------:|:----:|
| Baseline (zéro-shot) | 14,48 | — |
| FT, lr = 5×10⁻⁵ | 13,71 | −0,77 |
| FT, lr = 3×10⁻⁵ | 13,14 | −1,34 |
| FT, lr = 1×10⁻⁵ | 12,58 | −1,90 |
| FT, encodeur gelé | 13,50 | −0,98 |
| **FT + augmentation audio** | **11,87** | **−2,61** |

**Test final** (500 ex.) : WER = **13,41 %**, CER = **6,12 %**

---

## ⚙️ Configuration expérimentale

| Paramètre | Valeur |
|:----------|:-------|
| Modèles | `openai/whisper-small` · `openai/whisper-medium` · `openai/whisper-large-v3` |
| Dataset | `facebook/multilingual_librispeech` (fr) |
| Train / Val / Test | 2 000 / 300 / 500 exemples |
| GPU | Tesla T4 (15,6 Go VRAM) — Google Colab |
| Époques | 3 · Batch 16 · Warmup 500 pas · Beam b=5 |
| Augmentation | AddGaussianNoise · TimeStretch · PitchShift |
| Suivi | Weights & Biases (`inf8225-whisper-french`) |

---

## 🗂️ Artifacts WandB

Projet WandB : **`inf8225-whisper-french`**

| Artifact | Nom complet | Type | Fichier contenu |
|:---------|:------------|:----:|:----------------|
| Évaluation baseline | `inf8225-whisper-french-baseline` | `evaluation` | `baseline_results.json` |
| Fine-tuning | `inf8225-whisper-french-finetuning` | `model` | Checkpoint whisper-small |
| Ablation LR | `inf8225-whisper-french-ablation-lr` | `ablation` | `ablation_lr.json` |
| Ablation gel encodeur | `inf8225-whisper-french-ablation-freeze` | `ablation` | `ablation_freeze.json` |
| Ablation augmentation | `inf8225-whisper-french-ablation-augment` | `ablation` | `ablation_augment.json` |
| Figures finales | `inf8225-whisper-french-figures` | `figures` | Toutes les figures |

### Noms des runs WandB

| Run | Expérience |
|:----|:-----------|
| `baseline-evaluation` | Évaluation zéro-shot des 3 variantes |
| `fine-tuning-lr1e-05` | Fine-tuning small, lr=1e-5 **(config retenue)** |
| `ablation-lr` | Ablation taux d'apprentissage |
| `ablation-freeze` | Ablation gel de l'encodeur |
| `ablation-augmentation` | Fine-tuning + augmentation audio |
| `figures-final` | Export des figures |
| `ckpt-<step>` | Checkpoints intermédiaires |



## 📦 Modèles Hugging Face

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Forcer le français
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="french", task="transcribe"
)
```

---

## 📄 Rapport

Rapport complet (format IJCAI, 4 pages) dans le dossier `rapport/`.

```bash
cd rapport/
pdflatex rapport_INF8225.tex
pdflatex rapport_INF8225.tex   # second passage pour les références
```

**Overleaf** : importer `INF8225_Overleaf.zip` via *New Project → Upload Project*.

---

## 🔗 Liens

| Ressource | Lien |
|:----------|:-----|
| Paper Whisper | [arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356) |
| MLS Dataset | [HuggingFace — multilingual_librispeech](https://huggingface.co/datasets/facebook/multilingual_librispeech) |
| Whisper small | [HuggingFace — openai/whisper-small](https://huggingface.co/openai/whisper-small) |
| WandB project | [wandb.ai — inf8225-whisper-french](https://wandb.ai) |

---

## 👥 Équipe 12

| Nom | Email |
|:----|:------|
| Hamza El Haddad | hamza.el-haddad@polymtl.ca |
| Jules Gouy | jules.gouy@polymtl.ca |

*INF8225 — Techniques probabilistes et d'apprentissage en IA · École Polytechnique de Montréal · Avril 2026*
