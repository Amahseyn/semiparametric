"""
Relative Location of CT Slices Benchmark & Ablation Study
Evaluates on 5,000 observations with 384 predictive features.
Supports:
  - 5-Fold Cross-Validation
  - Model Ablation Decomposition (Linear, OLS-NN, Weighted-OLS, BPW-NN)
  - In-fold preprocessing pipeline and legacy modes
  - Variability reporting (Mean +/- Std)
"""

import argparse
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.data_loader import load_ct_slices_dataset, prepare_pca_features
from src.hybrid_model import HybridBPWNN
from src.metrics import print_metrics, aggregate_cv_results, print_ablation_table


def run_ct_slices_benchmark(strict_cv=True, limit=5000, epochs=50):
    print("=" * 75)
    mode_str = "In-Fold Preprocessing Pipeline" if strict_cv else "Standard PCA Pipeline"
    print(f"CT Slices Benchmark | Mode: {mode_str}")
    print("=" * 75)

    X_raw, y_raw = load_ct_slices_dataset(limit=limit)
    print(f"Dataset loaded: {X_raw.shape[0]} instances, {X_raw.shape[1]} features (patient ID excluded).")

    models_to_run = [
        "Linear Regression",
        "Ridge",
        "OLS-NN",
        "Ridge-NN",
        "Lasso-NN",
        "Elastic-NN",
        "W-OLS-NN",
        "RBPW-NN",
        "LBPW-NN",
        "EBPW-NN",
    ]

    model_metrics = {name: [] for name in models_to_run}
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    if not strict_cv:
        X_pca, y_scaled, _, scaler_y, _ = prepare_pca_features(X_raw, y_raw, n_components=4)
        X_param_all = X_pca[:, [0, 1, 2]]
        X_nonparam_all = X_pca[:, 3]

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_raw), 1):
        print(f"\n--- Fold {fold_idx} / 5 ---")

        if strict_cv:
            sc_x = StandardScaler()
            X_tr_sc = sc_x.fit_transform(X_raw[train_idx])
            X_va_sc = sc_x.transform(X_raw[val_idx])

            sc_y = StandardScaler()
            y_tr = sc_y.fit_transform(y_raw[train_idx].reshape(-1, 1)).ravel()
            y_va = sc_y.transform(y_raw[val_idx].reshape(-1, 1)).ravel()
            scaler_y_fold = sc_y

            pca = PCA(n_components=4)
            X_tr_pca = pca.fit_transform(X_tr_sc)
            X_va_pca = pca.transform(X_va_sc)

            X_tr_param, X_va_param = X_tr_pca[:, [0, 1, 2]], X_va_pca[:, [0, 1, 2]]
            X_tr_nonparam, X_va_nonparam = X_tr_pca[:, 3], X_va_pca[:, 3]
        else:
            X_tr_param, X_va_param = X_param_all[train_idx], X_param_all[val_idx]
            X_tr_nonparam, X_va_nonparam = X_nonparam_all[train_idx], X_nonparam_all[val_idx]
            y_tr, y_va = y_scaled[train_idx], y_scaled[val_idx]
            scaler_y_fold = scaler_y

        for model_name in models_to_run:
            hybrid = HybridBPWNN(
                model_type=model_name,
                hidden_layers=2,
                hidden_nodes=100,
                epochs=epochs,
                verbose=0,
            )
            y_pred_tr, y_pred_va, num_params = hybrid.fit_and_predict(
                X_train_param=X_tr_param,
                X_test_param=X_va_param,
                X_train_nonparam=X_tr_nonparam,
                X_test_nonparam=X_va_nonparam,
                y_train=y_tr,
            )
            metrics = print_metrics(
                y_train=y_tr,
                y_test=y_va,
                y_pred_train=y_pred_tr,
                y_pred_test=y_pred_va,
                model_name=model_name,
                num_params=num_params,
                scaler_y=scaler_y_fold,
            )
            model_metrics[model_name].append(metrics)

    aggregate_cv_results(model_metrics)
    print_ablation_table(model_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CT Slices Benchmark")
    parser.add_argument("--legacy", action="store_true", help="Run in legacy mode (global PCA before split)")
    parser.add_argument("--limit", type=int, default=5000, help="Number of observations to use (default 5000)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for training residual neural network")
    args = parser.parse_args()

    run_ct_slices_benchmark(strict_cv=(not args.legacy), limit=args.limit, epochs=args.epochs)
