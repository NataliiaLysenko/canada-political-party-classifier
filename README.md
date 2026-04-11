# canada-political-party-classifier
STAT 413 — Final Project, Winter 2026


Binary classification of Canadian federal ridings as Liberal vs. non-Liberal using soft-margin Support Vector Machines with linear and nonlinear kernels, trained via three gradient-based optimization algorithms: Robbins–Monro (SGD), AdaGrad, and Adam.

## How to Run 

```bash
git clone <repo-url>
cd canada-political-party-classifier
pip install -r requirements.txt
```
All experiment results are stored in the notebook output cells under `notebooks/`. To reproduce them, open any notebook and run all cells.

---

## Source Files

| File | Description |
|---|---|
| **`optimizers.py`** | **Main module**. Implements `robbins_monro_svm`, `adagrad_svm`, and `adam_svm` — all working on a kernelized soft-margin SVM in dual form. Also provides `tune_optimizer_joint_cv` for joint hyperparameter + kernel grid search with 10-fold stratified CV, and `repeated_eval_comparison` for multi-seed evaluation. |
| **`kernels.py`** | Defines four kernel functions (linear, RBF, polynomial, sigmoid) with a registry of default parameters and CV search grids.  |
| **`cv_tuning.py`** | Data preparation (load → clean → split → scale/encode → convert to SVM labels). Also defines the CV configuration. |
| **`helpers.py`** | Low-level methods shared across all .py files: data loading, target binarization, column dropping, feature lists, province-to-region mapping, and preprocessing (StandardScaler + OneHotEncoder). |
| **`preprocessing.py`** | Early script for data exploration — **not used in the final model**. Kept for reference. |

---


## Notebooks
- **`optimizers.ipynb`** — MAIN NOTEBOOK: Three-way comparison between optimizers. It contains: convergence curves, fit times, repeated-split evaluation across 10 random seeds. All optimizers achieve ~82% mean test accuracy.
- **`eda.ipynb`** — Exploratory analysis of the data (feature distributions, correlation analysis..).
- **`cv_tuning.ipynb`** — Python Sklearn (ML library) baselines for reference: Logistic Regression (~79.5% CV) and tuned SVC.
- **`RM.ipynb`** — Full Robbins–Monro run, also comparison with the experimental model with `Region` variable. **Not used in the final model**. Kept for reference.
- **`AdaGrad.ipynb`** — AdaGrad tuning and evaluation for experimental model with `Region` variable. **Not used in the final model**. Kept for reference.
- **`Adam.ipynb`** — Adam tuning and evaluation for experimental model with `Region` variable. **Not used in the final model**. Kept for reference.



## Data

The dataset (`data/ridings.csv`) contains 343 rows, one per Canadian federal riding, with census features (age, income, education, unemployment, etc.) and election outcomes. The target is binary: Liberal (170 ridings) vs. non-Liberal (173 ridings).

**Source:** Statistics Canada for Census Data, Elections Canada for Political Data
