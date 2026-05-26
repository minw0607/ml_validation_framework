# ML Model Validation Framework

<p align="center">
  <a href="https://colab.research.google.com/github/minw0607/ml_validation_framework/blob/main/notebooks/ML_Model_Validation.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python 3.8+"/>
  &nbsp;
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Validation-Classification%20%7C%20Regression-orange" alt="Task types"/>
</p>

<p align="center">
  <b>A comprehensive, interactive framework for validating machine learning models —<br/>
  from data quality through explainability, robustness, and fairness.</b>
</p>

---

## Why This Framework?

Training a model is the beginning, not the end. Regulatory frameworks (SR 11-7, SS1/23, TRIM), internal model risk governance, and responsible AI standards all require rigorous **model validation** before deployment — and ongoing monitoring thereafter.

This framework provides a structured, widget-driven validation workflow that covers every major dimension of model quality in one notebook. No configuration files, no boilerplate — select a dataset, follow the steps, and get a complete validation picture.

**Ideal for:**
- 🏦 Risk and compliance practitioners validating credit, fraud, or underwriting models
- 🔬 Data scientists conducting pre-deployment model audits
- 🎓 Researchers benchmarking and explaining ML models
- 👩‍💻 ML engineers building internal model governance tooling

---

## What's Covered

### Validation Dimensions

| Dimension | What It Checks |
|-----------|---------------|
| **Data Quality** | Missing values, distributions, outliers, feature correlations |
| **Feature Selection** | Pearson & Spearman correlation, permutation-based importance with threshold control |
| **Model Performance** | Side-by-side comparison across multiple algorithms and metrics |
| **Explainability — Global** | SHAP summary plots, permutation importance, Partial Dependence Plots (PDP) |
| **Explainability — Local** | SHAP waterfall charts and LIME explanations per instance |
| **Calibration & Reliability** | Reliability diagrams, conformal prediction (classification & regression) |
| **Overfit Detection** | Train-vs-test performance gaps segmented by decision tree leaves or feature bins |
| **Weak Spot Analysis** | Identifies data segments where the model underperforms vs the baseline |
| **Resiliency** | Worst-sample and worst-cluster analysis; PSI, Wasserstein, and KS distance measures |
| **Robustness** | Sensitivity to input perturbations across feature subspaces |
| **Fairness** | Group-level performance comparison across protected or demographic attributes |

### Supported Models

| Model | Classification | Regression |
|-------|:--------------:|:----------:|
| Logistic Regression | ✅ | — |
| Random Forest | ✅ | ✅ |
| Gradient Boosting Machine (GBM) | ✅ | ✅ |
| LightGBM | ✅ | ✅ |
| Multi-Layer Perceptron (MLP) | ✅ | ✅ |

> All models support **configurable hyperparameters** directly from the notebook UI — no code changes needed.

### Metrics

| Task | Metrics |
|------|---------|
| **Classification** | AUC-ROC, Accuracy, Precision, Recall, F1 Score |
| **Regression** | MSE, RMSE, MAE, R² |

---

## Workflow

The notebook guides you through **10 sequential steps**, each powered by interactive widgets:

```
  Load Data  →  Summary  →  Info  →  Preprocess  →  Model Setup
                                                          │
  Diagnose  ←  Explain  ←  Register  ←  Train  ←  Feature Select
```

| Step | Method | Description |
|------|--------|-------------|
| 1 | `data_loader()` | Select a built-in dataset or upload your own CSV |
| 2 | `data_summary()` | Distributions, correlation heatmap, missing-value analysis |
| 3 | `data_info()` | Schema: column names, types, record counts |
| 4 | `data_preprocess()` | Interactive imputation and normalization |
| 5 | `model_prepare()` | Set target variable, task type, and train/test split |
| 6 | `feature_select()` | Pearson & Spearman correlation, permutation importance |
| 7 | `model_training()` | Compare multiple algorithms with tunable hyperparameters |
| 8 | `model_register()` | Register best model(s) for downstream diagnostics |
| 9 | `model_explain()` | Global and local explainability (SHAP, LIME, PDP) |
| 10 | `model_diagnose()` | Full diagnostic suite: calibration, weak spots, resiliency, fairness |

---

## Quick Start — Google Colab

**No installation required.** Click below and run all cells:

<p align="center">
  <a href="https://colab.research.google.com/github/minw0607/ml_validation_framework/blob/main/notebooks/ML_Model_Validation.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="32"/>
  </a>
</p>

1. **Runtime → Run all** (or `Ctrl+F9`)
2. Select a dataset from the dropdown — built-in datasets auto-configure task type and target variable
3. Follow each step in order

> The setup cell clones this repo automatically, so `src/` and `data/` are always available without any manual uploads.

---

## Local Setup

```bash
git clone https://github.com/minw0607/ml_validation_framework.git
cd ml_validation_framework
pip install -r requirements.txt
jupyter notebook notebooks/ML_Model_Validation.ipynb
```

Requirements: Python 3.8+, see [`requirements.txt`](requirements.txt) for full dependency list.

---

## Built-in Datasets

Seven datasets are preloaded and ready to use — no downloads, no file paths to configure:

| Dataset | Task | Rows | Features | Domain |
|---------|------|-----:|--------:|--------|
| **Breast Cancer** (sklearn) | Binary classification | 569 | 30 | Healthcare |
| **Wine Quality** (sklearn) | Multi-class classification | 178 | 13 | Food science |
| **Iris** (sklearn) | Multi-class classification | 150 | 4 | Botany |
| **Diabetes** (sklearn) | Regression | 442 | 10 | Healthcare |
| **California Housing** (sklearn) | Regression | 20,640 | 8 | Real estate |
| **GMSC Credit Risk** | Binary classification | 30,000* | 10 | Credit risk |
| **Taiwan Credit Default** | Binary classification | 30,000 | 23 | Credit risk |

\* Sampled from 150 k rows for interactive performance; full dataset in `data/cs-training.csv`.

You can also load **any CSV** via the Upload option in Step 1.

---

## Repo Structure

```
ml_validation_framework/
├── notebooks/
│   └── ML_Model_Validation.ipynb   # Lightweight demo — imports and step calls only
├── src/
│   ├── __init__.py
│   └── validation.py               # ValidationFramework class — all logic lives here
├── data/
│   ├── README.md                   # Dataset descriptions and sources
│   ├── cs-training.csv             # GMSC Credit Risk dataset
│   └── UCI_Credit_Card.csv         # Taiwan Credit Default dataset
├── requirements.txt
├── LICENSE
└── README.md
```

The notebook is intentionally thin — it imports `ValidationFramework` from `src/validation.py` and calls one method per step. All widget logic, model code, and visualization is encapsulated in the source class, making it easy to extend or integrate into other projects.

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | `pandas`, `numpy` |
| Models | `scikit-learn`, `lightgbm` |
| Explainability | `shap`, `lime`, `eli5` |
| Visualization | `matplotlib`, `seaborn`, `graphviz` |
| UI | `ipywidgets` |
| Statistics | `scipy` |

---

## License

MIT — see [`LICENSE`](LICENSE) for details.
