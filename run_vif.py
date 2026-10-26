"""
VIF Diagnostic Analysis Script for Simulation, Concrete, and Energy Datasets.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from src.data_loader import (
    generate_simulation_data,
    load_concrete_dataset,
    load_energy_dataset,
)


def compute_vif_table(X, feature_names=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cols = feature_names if feature_names else [f"X{i+1}" for i in range(X.shape[1])]

    if HAS_STATSMODELS:
        vif_vals = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    else:
        # Fallback using inverse correlation matrix: VIF_j = [corr(X)^(-1)]_jj
        corr = np.corrcoef(X_scaled, rowvar=False)
        try:
            inv_corr = np.linalg.inv(corr)
            vif_vals = np.diag(inv_corr)
        except np.linalg.LinAlgError:
            # Handle singular matrix (e.g. perfect multicollinearity -> inf)
            inv_corr = np.linalg.pinv(corr)
            vif_vals = [np.inf if abs(np.linalg.det(corr)) < 1e-10 else v for v in np.diag(inv_corr)]

    return pd.DataFrame({"Feature": cols, "VIF": vif_vals})


def main():
    print("=" * 60)
    print("Variance Inflation Factor (VIF) Analysis")
    print("=" * 60)

    # 1. Simulation
    print("\n--- 1. Simulation Dataset (Table 1) ---")
    X_sim, _, _ = generate_simulation_data(n=500, seed=42)
    vif_sim = compute_vif_table(X_sim, ["X1", "X2", "X3", "X4"])
    print(vif_sim.to_string(index=False))

    # 2. Concrete
    print("\n--- 2. Concrete Compressive Strength Dataset (Table 3) ---")
    X_conc, _, conc_names = load_concrete_dataset()
    vif_conc = compute_vif_table(X_conc, conc_names)
    print(vif_conc.to_string(index=False))

    # 3. Energy
    print("\n--- 3. Energy Efficiency Dataset (Table 6) ---")
    X_energy, _ = load_energy_dataset("data/enb2012.xlsx")
    energy_names = [f"X{i+1}" for i in range(X_energy.shape[1])]
    vif_energy = compute_vif_table(X_energy, energy_names)
    print(vif_energy.to_string(index=False))


if __name__ == "__main__":
    main()
