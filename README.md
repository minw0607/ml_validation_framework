# ML Model Validation Framework

An interactive, widget-driven framework for end-to-end machine learning model validation —
built for risk and compliance practitioners, data scientists, and ML engineers.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/ml-model-validation/blob/main/notebooks/ML_Model_Validation.ipynb)

---

## What It Does

The framework guides you through ten validation steps via interactive widgets — no code
changes needed after setup:

| Step | Description |
|------|-------------|
| **Data Loading** | 7 preloaded datasets or upload your own CSV |
| **Data Summary** | Distributions, correlation heatmap, missing-value analysis |
| **Data Info** | Schema, dtypes, record counts |
| **Preprocessing** | Interactive imputation and normalization |
| **Model Setup** | Target variable, task type, train/test split ratio |
| **Feature Selection** | Pearson & Spearman correlation, permutation importance |
| **Model Training** | Logistic Regression, Random Forest, GBM, LightGBM, MLP with configurable hyperparameters |
| **Model Registration** | Register best model(s) for downstream diagnostics |
| **Explainability** | Global (SHAP, permutation importance, PDP) and local (SHAP waterfall, LIME) |
| **Diagnostics** | Accuracy, calibration (conformal prediction), overfit, weak spots, resiliency, robustness, fairness |

---

## Quick Start — Google Colab (Recommended)

Click the **Open in Colab** badge above, then:

1. **Runtime → Run all** (or `Ctrl+F9`)
2. Select a dataset from the dropdown
3. Follow the steps in order — each widget builds on the previous one

No API keys or local setup required.

---

## Local Setup

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/ml-model-validation.git
cd ml-model-validation
pip install -r requirements.txt
jupyter notebook notebooks/ML_Model_Validation.ipynb
```

---

## Built-in Datasets

| Dataset | Task | Rows | Features |
|---------|------|------|----------|
| Breast Cancer (sklearn) | Binary classification | 569 | 30 |
| Wine Quality (sklearn) | Multi-class classification | 178 | 13 |
| Iris (sklearn) | Multi-class classification | 150 | 4 |
| Diabetes (sklearn) | Regression | 442 | 10 |
| California Housing (sklearn) | Regression | 20,640 | 8 |
| GMSC Credit Risk | Binary classification | 30,000* | 10 |
| Taiwan Credit Default | Binary classification | 30,000 | 23 |

\* Sampled from 150 k for interactive speed; full file in `data/cs-training.csv`.

You can also upload any CSV file directly via the **Upload CSV** option.

---

## Repo Structure

```
ml-model-validation/
├── notebooks/
│   └── ML_Model_Validation.ipynb   # lightweight demo notebook (import + run)
├── src/
│   ├── __init__.py
│   └── validation.py               # ValidationFramework class (all logic)
├── data/
│   ├── README.md
│   ├── cs-training.csv             # GMSC Credit Risk
│   └── UCI_Credit_Card.csv         # Taiwan Credit Default
├── requirements.txt
└── README.md
```

---

## License

MIT
