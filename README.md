# CMR Report Multi-Label Classifier

Code companion to the paper:

> **[Paper title]** — [Authors], [Journal/Conference], [Year].

This repository contains the refactored analysis code accompanying our Radiology: AI paper, released to support reproducibility..
Patient data is **not** included.

---

## Overview

We fine-tune [SciBERT](https://huggingface.co/allenai/scibert_scivocab_cased) on
Cardiac MRI (CMR) radiology reports to predict six diagnostic labels simultaneously:

| Label  | Meaning |
|--------|---------|
| Normal | No significant finding |
| DCM    | Dilated cardiomyopathy |
| HCM    | Hypertrophic cardiomyopathy |
| LADF   | Left anterior descending artery finding |
| RCAF   | Right coronary artery finding |
| LCXF   | Left circumflex artery finding |

Two model variants are trained independently:

- **Content model** — trained on the full report body (max 512 tokens, 3 epochs).
- **Conclusion model** — trained on the conclusion section only (max 400 tokens, 5 epochs).

Final predictions are obtained by ensembling (averaging) the two models' output probabilities.

---

## Method Summary

### Model architecture

```
SciBERT encoder (allenai/scibert_scivocab_cased)
    ↓  [CLS] pooled output
Dropout(p=0.5)
    ↓
Linear(768 → 6)   ← Xavier uniform initialisation
    ↓
BCEWithLogitsLoss (sigmoid applied internally)
```

### Training details

| Setting | Content model | Conclusion model |
|---------|--------------|-----------------|
| Epochs | 3 | 5 |
| Batch size | 2 | 2 |
| Max sequence length | 512 | 400 |
| Learning rate | 1e-5 | 1e-5 |
| Optimiser | AdamW | AdamW |
| Weight decay (weights) | 0.15 | 0.15 |
| Weight decay (bias/LN) | 0.0 | 0.0 |
| LR schedule | Linear warmup | Linear warmup |
| Gradient clipping | max_norm=1.0 | max_norm=1.0 |
| Dropout | 0.5 | 0.5 |
| Random seed | 13 | 13 |

### Class imbalance

Multi-label datasets are highly imbalanced.
We compute per-class positive/negative frequencies on the training split and
pass the resulting `pos_weight` tensor to `BCEWithLogitsLoss`.

### Data split

| Split | Size |
|-------|------|
| Train | 80 % |
| Validation | 10 % |
| Test | 10 % |

Stratified on the DCM column to preserve class proportions.
Best model checkpoint is selected based on **validation F1-micro**.

### Interpretability

Gradient-based token saliency (`src/utils.py: compute_token_saliency`) highlights
which words in a report most influenced each prediction, supporting clinical transparency.

---

## Data

Patient-level CMR reports are **not** included in this repository because they
constitute protected health information.

The code expects two CSV files with the following structure:

```
data/raw/CMR_labels_w_missing.csv          ← full-report variant
data/raw/Conclusions_labels_w_missing_modified.csv  ← conclusion variant
```

**Expected CSV schema:**

| Column | Type | Description |
|--------|------|-------------|
| Content | str | Full radiology report text |
| Conclusion | str | Conclusion section of the report |
| Normal | int (0/1) | Label |
| DCM | int (0/1) | Label |
| HCM | int (0/1) | Label |
| LADF | int (0/1) | Label |
| RCAF | int (0/1) | Label |
| LCXF | int (0/1) | Label |

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. A GPU is strongly advised for fine-tuning SciBERT.

---

## Usage

### Train both model variants

```bash
python train.py --config config/config.yaml --variant both
```

### Train a single variant

```bash
python train.py --config config/config.yaml --variant content
python train.py --config config/config.yaml --variant conclusion
```

All hyperparameters are controlled through [config/config.yaml](config/config.yaml).
Checkpoints are saved to `checkpoints/` (excluded from git).

---

## Repository structure

```
.
├── config/
│   └── config.yaml        # All hyperparameters and paths
├── src/
│   ├── dataset.py         # CMRDataset + make_data_loader
│   ├── model.py           # DiagnosticClassifier (SciBERT head)
│   ├── train.py           # train_epoch, eval_model, run_training
│   ├── evaluate.py        # ROC, calibration, history plots
│   └── utils.py           # Seed, class weights, saliency
├── train.py               # CLI entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{Zaman2022,
  title = {Automatic Diagnosis Labeling of Cardiovascular MRI by Using Semisupervised Natural Language Processing of Text Reports},
  volume = {4},
  ISSN = {2638-6100},
  url = {http://dx.doi.org/10.1148/ryai.210085},
  DOI = {10.1148/ryai.210085},
  number = {1},
  journal = {Radiology: Artificial Intelligence},
  publisher = {Radiological Society of North America (RSNA)},
  author = {Zaman,  Sameer and Petri,  Camille and Vimalesvaran,  Kavitha and Howard,  James and Bharath,  Anil and Francis,  Darrel and Peters,  Nicholas and Cole,  Graham D. and Linton,  Nick},
  year = {2022},
  month = jan 
}
```

---

## License

[MIT](LICENSE) — code only. Patient data is not part of this release.
