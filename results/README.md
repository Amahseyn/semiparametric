# Benchmark Results Summary

This directory provides aggregated empirical metrics and component-ablation summaries obtained across all simulation experiments and real-world benchmark datasets.

All evaluations are conducted under a strict in-fold preprocessing pipeline (leak-free standardization and PCA within training folds).

---

## 1. Simulation Study (100 Monte Carlo Repetitions)

* **Data Generation:** $n = 500$, $p = 4$ predictors with high collinearity ($\rho = 0.9$ between $X_1$ and $X_2$).
* **Evaluation:** 100 independent Monte Carlo test runs.

### Variability Analysis (Mean ± Std Across 100 Reps)
| Model | Test MSE (Mean ± Std) | Test R² (Mean ± Std) | Mean AIC | Mean BIC |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | $3.417 \pm 0.486$ | $0.464 \pm 0.079$ | 129.85 | 140.27 |
| **Ridge** | $3.417 \pm 0.486$ | $0.464 \pm 0.079$ | 129.86 | 140.28 |
| **OLS-NN** | $1.549 \pm 0.262$ | $0.757 \pm 0.044$ | 752.30 | 1677.13 |
| **Ridge-NN** | $1.546 \pm 0.265$ | $0.757 \pm 0.043$ | 752.10 | 1676.93 |
| **Lasso-NN** | $1.551 \pm 0.265$ | $0.756 \pm 0.045$ | 752.46 | 1677.30 |
| **Elastic-NN** | $1.552 \pm 0.261$ | $0.756 \pm 0.045$ | 752.52 | 1677.35 |
| **W-OLS-NN** | $1.692 \pm 0.318$ | $0.734 \pm 0.052$ | 760.87 | 1685.70 |
| **RBPW-NN** | $1.695 \pm 0.320$ | $0.734 \pm 0.053$ | 760.98 | 1685.82 |
| **LBPW-NN** | $1.693 \pm 0.321$ | $0.734 \pm 0.052$ | 760.87 | 1685.71 |
| **EBPW-NN** | $1.691 \pm 0.319$ | $0.734 \pm 0.053$ | 760.81 | 1685.64 |

### Component Ablation Study
| Model Configuration | Residual NN | Weighting $W$ | Target Shrinkage $\boldsymbol{\Lambda}$ | Test MSE |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | Parametric | No ($W=I$) | No ($\boldsymbol{\Lambda}=0$) | 3.417 |
| **Ridge** | Parametric | No ($W=I$) | Ridge ($\boldsymbol{\beta}_0=0$) | 3.417 |
| **OLS-NN** | Neural Network | No ($W=I$) | No ($\boldsymbol{\Lambda}=0$) | 1.549 |
| **Ridge-NN** | Neural Network | No ($W=I$) | Ridge ($\boldsymbol{\beta}_0=0$) | 1.546 |
| **W-OLS-NN** | Neural Network | Yes ($W=S^{-1}$) | No ($\boldsymbol{\Lambda}=0$) | 1.692 |
| **RBPW-NN** | Neural Network | Yes ($W=S^{-1}$) | Ridge ($\boldsymbol{\beta}_0=\hat{\boldsymbol{\beta}}_R$) | 1.695 |
| **LBPW-NN** | Neural Network | Yes ($W=S^{-1}$) | Lasso ($\boldsymbol{\beta}_0=\hat{\boldsymbol{\beta}}_L$) | 1.693 |
| **EBPW-NN** | Neural Network | Yes ($W=S^{-1}$) | Elastic ($\boldsymbol{\beta}_0=\hat{\boldsymbol{\beta}}_E$) | 1.691 |

---

## 2. Concrete Compressive Strength Benchmark (5-Fold CV)

* **Dataset:** 1,030 instances, 8 physical components. Target: Compressive Strength (MPa).
* **Evaluation:** 5-fold cross-validation with in-fold scaling and PCA.

### Variability Analysis (Mean ± Std Across Folds)
| Model | Test MSE (Mean ± Std) | Test R² (Mean ± Std) | Mean AIC | Mean BIC |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | $187.616 \pm 21.691$ | $0.325 \pm 0.044$ | 1084.84 | 1098.16 |
| **Ridge** | $187.615 \pm 21.673$ | $0.326 \pm 0.043$ | 1084.85 | 1098.16 |
| **OLS-NN** | $194.036 \pm 22.261$ | $0.302 \pm 0.047$ | 2493.86 | 4840.02 |
| **Ridge-NN** | $198.168 \pm 28.333$ | $0.288 \pm 0.074$ | 2497.47 | 4843.62 |
| **Lasso-NN** | $194.979 \pm 28.494$ | $0.302 \pm 0.040$ | 2494.06 | 4840.21 |
| **Elastic-NN** | $192.723 \pm 17.397$ | $0.306 \pm 0.034$ | 2493.00 | 4839.15 |
| **W-OLS-NN** | $193.987 \pm 20.409$ | $0.301 \pm 0.051$ | 2494.03 | 4840.19 |
| **RBPW-NN** | $197.237 \pm 24.403$ | $0.291 \pm 0.046$ | 2497.04 | 4843.19 |
| **LBPW-NN** | $199.502 \pm 16.186$ | $0.279 \pm 0.066$ | 2500.27 | 4846.42 |
| **EBPW-NN** | $203.413 \pm 27.930$ | $0.270 \pm 0.048$ | 2503.09 | 4849.24 |

---

## 3. Relative Location of CT Slices Benchmark (5-Fold CV)

* **Dataset:** 5,000 instances, 384 continuous image-derived histogram features (`patientId` excluded). Target: Axial Slice Location.
* **Evaluation:** 5-fold cross-validation with in-fold scaling and PCA.

### Variability Analysis (Mean ± Std Across Folds)
| Model | Test MSE (Mean ± Std) | Test R² (Mean ± Std) | Mean AIC | Mean BIC |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | $208.802 \pm 6.361$ | $0.388 \pm 0.017$ | 5348.91 | 5368.54 |
| **Ridge** | $208.802 \pm 6.362$ | $0.388 \pm 0.017$ | 5348.91 | 5368.54 |
| **OLS-NN** | $198.249 \pm 8.257$ | $0.419 \pm 0.019$ | 27698.64 | 82690.04 |
| **Ridge-NN** | $\mathbf{196.932 \pm 8.262}$ | $\mathbf{0.423 \pm 0.022}$ | 27691.97 | 82683.36 |
| **Lasso-NN** | $203.225 \pm 11.192$ | $0.405 \pm 0.020$ | 27722.71 | 82714.11 |
| **Elastic-NN** | $203.331 \pm 10.388$ | $0.404 \pm 0.019$ | 27723.46 | 82714.86 |
| **W-OLS-NN** | $204.212 \pm 6.793$ | $0.401 \pm 0.016$ | 27728.59 | 82719.98 |
| **RBPW-NN** | $201.428 \pm 6.295$ | $0.409 \pm 0.015$ | 27714.94 | 82706.33 |
| **LBPW-NN** | $200.369 \pm 6.635$ | $0.412 \pm 0.020$ | 27709.61 | 82701.01 |
| **EBPW-NN** | $205.682 \pm 7.214$ | $0.397 \pm 0.016$ | 27735.71 | 82727.11 |

---

## 4. Energy Efficiency Benchmark (5-Fold CV)

* **Dataset:** 768 building configurations, 8 architectural features. Target: Cooling Load (kWh/m²).
* **Evaluation:** 5-fold cross-validation with in-fold scaling and PCA.

### Variability Analysis (Mean ± Std Across Folds)
| Model | Test MSE (Mean ± Std) | Test R² (Mean ± Std) | Mean AIC | Mean BIC |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | $15.884 \pm 2.033$ | $0.823 \pm 0.017$ | 431.61 | 443.75 |
| **Ridge** | $15.885 \pm 2.035$ | $0.823 \pm 0.017$ | 431.62 | 443.76 |
| **OLS-NN** | $23.789 \pm 9.645$ | $0.734 \pm 0.111$ | 22886.24 | 56886.09 |
| **Ridge-NN** | $16.931 \pm 2.108$ | $0.812 \pm 0.017$ | 22843.48 | 56843.34 |
| **Lasso-NN** | $20.425 \pm 6.161$ | $0.775 \pm 0.056$ | 22867.12 | 56866.97 |
| **Elastic-NN** | $17.411 \pm 2.624$ | $0.806 \pm 0.023$ | 22847.31 | 56847.17 |
| **W-OLS-NN** | $17.582 \pm 2.701$ | $0.804 \pm 0.024$ | 22848.67 | 56848.52 |
| **RBPW-NN** | $17.340 \pm 1.999$ | $0.806 \pm 0.028$ | 22847.21 | 56847.07 |
| **LBPW-NN** | $20.787 \pm 4.091$ | $0.768 \pm 0.045$ | 22873.24 | 56873.09 |
| **EBPW-NN** | $\mathbf{16.298 \pm 1.510}$ | $\mathbf{0.818 \pm 0.015}$ | 22838.07 | 56837.93 |

### Component Ablation Study (Energy Efficiency)
| Model Configuration | Residual NN | Weighting $W$ | Target Shrinkage $\boldsymbol{\Lambda}$ | Test MSE |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | Parametric | No ($W=I$) | No ($\boldsymbol{\Lambda}=0$) | 15.884 |
| **Ridge** | Parametric | No ($W=I$) | Ridge ($\boldsymbol{\beta}_0=0$) | 15.885 |
| **OLS-NN** | Neural Network | No ($W=I$) | No ($\boldsymbol{\Lambda}=0$) | **23.789** (Severe Overfitting) |
| **Ridge-NN** | Neural Network | No ($W=I$) | Ridge ($\boldsymbol{\beta}_0=0$) | 16.931 |
| **W-OLS-NN** | Neural Network | **Yes ($W=S^{-1}$)** | No ($\boldsymbol{\Lambda}=0$) | **17.582** (Weighting Stabilizer) |
| **RBPW-NN** | Neural Network | Yes ($W=S^{-1}$) | Ridge ($\boldsymbol{\beta}_0=\hat{\boldsymbol{\beta}}_R$) | 17.340 |
| **LBPW-NN** | Neural Network | Yes ($W=S^{-1}$) | Lasso ($\boldsymbol{\beta}_0=\hat{\boldsymbol{\beta}}_L$) | 20.787 |
| **EBPW-NN** | Neural Network | **Yes ($W=S^{-1}$)** | **Elastic ($\boldsymbol{\beta}_0=\hat{\boldsymbol{\beta}}_E$)** | **16.298** (Lowest Std: $\pm 1.510$) |
