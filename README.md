# Semi-Parametric Regression Model Combining BPW and Neural Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official reproducible Python implementation of the **Balanced Penalized Weighted (BPW)** semi-parametric regression framework coupled with deep residual neural networks.

---

## 📖 Overview

In complex regression environments with severe multicollinearity and mixed linear/non-linear signals, standard parametric methods (such as OLS, Ridge, Lasso, and Elastic Net) often suffer from coefficient instability or over-simplification, while pure deep neural networks lose coefficient interpretability.

This framework introduces a two-stage semi-parametric additive model:
$$y = X\boldsymbol{\beta} + f(Z) + \varepsilon$$

1. **Parametric BPW Component ($X\boldsymbol{\beta}$)**: Estimates linear coefficients using an adaptive, data-driven diagonal weighting matrix $W$ based on row-wise residual dispersion ($w_i = 1 / S_i$) and a data-driven prior regularization matrix $\boldsymbol{\Lambda} = \widehat{\mathrm{Cov}}(X)$:
   $$\hat{\boldsymbol{\beta}}_{\text{BPW}} = (\boldsymbol{\Lambda} + X^\top W X)^{-1} (\boldsymbol{\Lambda} \hat{\boldsymbol{\beta}}_P + X^\top W y)$$
   where $\hat{\boldsymbol{\beta}}_P$ is a proposal coefficient vector from Ridge, Lasso, or Elastic Net.
2. **Non-Parametric Residual Neural Network ($f(Z)$)**: A feedforward neural network trained directly on the residuals $(y - X\hat{\boldsymbol{\beta}}_{\text{BPW}})$ to capture underlying non-linear structures.

---

## 📂 Repository Structure

```text
semiparametric/
├── README.md                          # Documentation and reproduction guide
├── requirements.txt                   # Python package dependencies
├── data/                              # Benchmark dataset storage
│   └── enb2012.xlsx                   # Energy Efficiency dataset
├── src/                               # Modular core library
│   ├── __init__.py                    # Package exports
│   ├── bpw.py                         # BPW closed-form estimator and proposals (Ridge, Lasso, Elastic Net)
│   ├── hybrid_model.py                # Hybrid semi-parametric learner (parametric + residual NN)
│   ├── metrics.py                     # Evaluation metrics (MSE, R^2, AIC, BIC) and fold aggregators
│   └── data_loader.py                 # Automated data fetchers, PCA preprocessing, and scaling
├── notebooks/                         # Clean, self-contained Jupyter notebooks
│   ├── 01_Simulation_Study.ipynb      # Table 2: 100 Monte Carlo simulation runs
│   ├── 02_Concrete_Compressive_Strength.ipynb # Table 4: 5-fold CV on Concrete benchmark
│   ├── 03_CT_Slices_Localization.ipynb       # Table 5: 5-fold CV on CT Slices benchmark (384 features)
│   ├── 04_Energy_Efficiency.ipynb            # Table 6: 5-fold CV on Energy Efficiency benchmark
│   └── 05_VIF_Analysis.ipynb                 # Diagnostic Variance Inflation Factor (VIF) tables
├── run_simulation.py                  # Standalone CLI script for Simulation study
├── run_concrete.py                    # Standalone CLI script for Concrete benchmark
├── run_ct_slices.py                   # Standalone CLI script for CT Slices benchmark
├── run_energy.py                      # Standalone CLI script for Energy Efficiency benchmark
└── run_vif.py                         # Standalone CLI script for VIF diagnostic tables
```

---

## 🚀 Installation & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Amahseyn/semiparametric.git
   cd semiparametric
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Replicating the Paper Results

You can reproduce all tables in the manuscript using either the command-line scripts or the interactive Jupyter notebooks.

### 1. Terminal / Command-Line Execution
* **Simulation Study (Table 2):**
  ```bash
  python run_simulation.py --reps 100 --samples 500 --epochs 100
  ```
* **Concrete Compressive Strength Benchmark (Table 4):**
  ```bash
  python run_concrete.py
  ```
* **CT Slices Localization Benchmark (Table 5):**
  ```bash
  python run_ct_slices.py
  ```
* **Energy Efficiency Benchmark (Table 6):**
  ```bash
  python run_energy.py
  ```
* **VIF Diagnostic Tables (Tables 1, 3, 6):**
  ```bash
  python run_vif.py
  ```

### 2. Interactive Jupyter Notebooks
Launch Jupyter Lab or Notebook:
```bash
jupyter lab notebooks/
```
Run any of the notebooks `01` through `05` sequentially.

---

## ⚙️ Hyperparameter Configuration Summary

All experiments use the exact settings detailed in Table 1 of the manuscript:

| Parameter | Simulation | Concrete | CT Slices | Energy Efficiency |
| :--- | :---: | :---: | :---: | :---: |
| **Hidden Layers** | 1 | 1 | 2 | 2 |
| **Neurons / Layer** | 50 | 100 | 100 | 100 |
| **Activation** | ReLU | ReLU | ReLU | ReLU |
| **Normalization** | Batch Normalization | Batch Normalization | Batch Normalization | Batch Normalization |
| **Optimizer** | Adam ($\text{lr} = 10^{-3}$) | Adam ($\text{lr} = 10^{-3}$) | Adam ($\text{lr} = 10^{-3}$) | Adam ($\text{lr} = 10^{-3}$) |
| **Epochs** | 100 | 50 | 50 | 50 |
| **Batch Size** | 32 | 32 | 32 | 32 |
| **Validation Scheme** | 80/20 Monte Carlo (100 runs) | 5-Fold Cross-Validation | 5-Fold Cross-Validation | 5-Fold Cross-Validation |
| **Retained PCs** | N/A (Orthogonal features) | 4 | 4 | 4 |

---

## 🔬 Benchmark Dataset Notes

1. **Concrete Compressive Strength (UCI ID 165):**
   1,030 instances, 8 physical components. Target: compressive strength (MPa).
2. **Relative Location of CT Slices (UCI ID 206):**
   - 384 image-derived polar-histogram features (`patientId` in column 0 is strictly excluded).
   - Evaluated on a representative subset of the first 5,000 observations. This prevents memory exhaustion ($53,500 \times 53,500$ dense matrix allocation requires $\approx 22.9$ GB RAM).
3. **Energy Efficiency (UCI ID 242 / `enb2012.xlsx`):**
   768 simulated residential building configurations, 8 architectural features. Target: Cooling Load (kWh/m²).

---

## 📄 Citation

```bibtex
@article{karamikabir2026bpw,
  title={Semi-Parametric Regression Model Combining BPW and Neural Networks},
  author={Karamikabir, Hamid and colleagues},
  journal={Journal of Statistical Computation and Simulation / Springer Nature},
  year={2026},
  doi={10.5281/zenodo.XXXXXXX}
}
```

## ⚖️ License
This project is open-source and licensed under the [MIT License](LICENSE).
