# INF8225 — Projet Final — Équipe 12
## Analyse Architecturale de Whisper pour l'ASR en Français

**Hamza El Haddad · Jules Gouy**  
École Polytechnique de Montréal — Avril 2026

---

## 📋 Description du projet

Ce projet réalise une étude architecturale systématique de [Whisper](https://github.com/openai/whisper) (OpenAI) appliquée à la reconnaissance automatique de la parole (ASR) en français sur le corpus **MLS French** (Multilingual LibriSpeech).

L'analyse porte sur trois axes :
1. **Scaling laws** — WER vs. nombre de paramètres (small / medium / large-v3)
2. **Compromis vitesse/précision** — latence d'inférence vs. WER
3. **Fine-tuning avec données limitées** — ablations sur le LR, le gel de l'encodeur et l'augmentation audio

---

## 📁 Structure du dépôt

```
.
├── README.md
├── INF8225_Whisper_Analysis_v3.ipynb   # Notebook principal
├── rapport/
│   ├── rapport_INF8225.tex             # Rapport LaTeX (format IJCAI)
│   ├── ijcai19.sty                     # Style IJCAI
│   ├── fig_losses.png                  # Courbes train/eval loss (WandB)
│   └── eval_wer_only.png               # Courbe eval/WER (WandB)
└── slides/
    └── INF8225_Equipe12_Slides_v2.pptx # Diapositives de présentation
```

---

## 🚀 Mise en route

### Prérequis

```bash
pip install transformers datasets accelerate
pip install jiwer evaluate
pip install wandb
pip install audiomentations
pip install torch torchaudio
```

### Exécution

Ouvrir le notebook sur **Google Colab** (recommandé — GPU Tesla T4) :

1. Monter Google Drive si besoin
2. Connecter WandB :  
   ```python
   import wandb
   wandb.login()
   ```
3. Exécuter les cellules dans l'ordre

> ⚠️ Les cellules de fine-tuning nécessitent un GPU. Sans GPU, seule l'évaluation zéro-shot est possible.

---

## 📊 Résultats principaux

### Scaling laws — Évaluation zéro-shot (MLS French, 500 ex.)

| Variante     | Paramètres | WER (%) | CER (%) | Latence (ms) |
|:-------------|:----------:|:-------:|:-------:|:------------:|
| small        | 244 M      | 14,48   | 6,55    | 1 220        |
| medium       | 769 M      |  9,38   | 4,05    | 2 740        |
| large-v3     | 1 550 M    |  6,17   | 3,99    | 3 063        |

### Fine-tuning de Whisper small (2 000 ex., MLS French)

| Configuration               | WER val (%) | Δ pp     |
|:----------------------------|:-----------:|:--------:|
| Baseline (zéro-shot)        | 14,48       | —        |
| FT, lr = 1e-5               | 12,58       | −1,90    |
| FT, lr = 3e-5               | 13,14       | −1,34    |
| FT, lr = 5e-5               | 13,71       | −0,77    |
| FT, encodeur gelé           | 13,50       | −0,98    |
| **FT + augmentation audio** | **11,87**   | **−2,61**|

**Test final** (500 ex.) : WER = **13,41 %**, CER = **6,12 %** (small, lr=1e-5)

---

## ⚙️ Configuration expérimentale

| Paramètre          | Valeur                              |
|:-------------------|:------------------------------------|
| Modèle             | `openai/whisper-small` (et variantes) |
| Dataset            | MLS French (Multilingual LibriSpeech) |
| Train / Val / Test | 2 000 / 300 / 500 exemples          |
| GPU                | Tesla T4 (15,6 Go VRAM)             |
| Époques            | 3                                   |
| Batch size         | 16                                  |
| Optimizer          | AdamW                               |
| Warmup steps       | 500                                 |
| Beam search        | b = 5                               |
| Suivi              | Weights & Biases                    |

### Augmentation audio (`audiomentations`)
- `AddGaussianNoise` : σ ∈ [0.001, 0.01]
- `TimeStretch` : ratio ∈ [0.9, 1.1]
- `PitchShift` : ±2 demi-tons

---

## 📦 Modèles Hugging Face utilisés

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# ou "openai/whisper-medium" / "openai/whisper-large-v3"
```

---

## 🔗 Liens utiles

| Ressource | Lien |
|:----------|:-----|
| Paper Whisper | [arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356) |
| MLS Dataset | [huggingface.co/datasets/facebook/multilingual_librispeech](https://huggingface.co/datasets/facebook/multilingual_librispeech) |
| WandB runs | [wandb.ai/equipe12-inf8225](https://wandb.ai) |
| Hugging Face Whisper | [huggingface.co/openai/whisper-small](https://huggingface.co/openai/whisper-small) |

---

## 📄 Rapport

Le rapport complet est disponible au format LaTeX (IJCAI) dans le dossier `rapport/`.  
Pour compiler localement :

```bash
cd rapport/
pdflatex rapport_INF8225.tex
pdflatex rapport_INF8225.tex   # second passage pour les références
```

Pour éditer sur **Overleaf** : importer le `.zip` fourni (contient `.tex`, `.sty` et les figures).

---

## 👥 Équipe 12

| Nom | Email |
|:----|:------|
| Hamza El Haddad | hamza.el-haddad@polymtl.ca |
| Jules Gouy | jules.gouy@polymtl.ca |

*INF8225 — Techniques probabilistes et d'apprentissage en IA*  
*École Polytechnique de Montréal — Avril 2026*
