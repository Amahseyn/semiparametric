"""
Replication Script: Simulation Study (Table 2)
Supports Monte Carlo repetitions (default: 100).
"""

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_loader import generate_simulation_data, prepare_pca_features
from src.hybrid_model import HybridBPWNN
from src.metrics import calculate_metrics, aggregate_cv_results, print_ablation_table


def run_simulation(n_repetitions=100, n_samples=500, epochs=100):
    print("=" * 60)
    print(f"Running Simulation Benchmark: {n_repetitions} Monte Carlo Repetitions")
    print("=" * 60)

    models_to_run = [
        ("Linear Regression", "Linear Regression"),
        ("Ridge", "Ridge"),
        ("OLS-NN", "OLS-NN"),
        ("Ridge-NN", "Ridge-NN"),
        ("Lasso-NN", "Lasso-NN"),
        ("Elastic-NN", "Elastic-NN"),
        ("W-OLS-NN", "W-OLS-NN"),
        ("RBPW-NN", "RBPW-NN"),
        ("LBPW-NN", "LBPW-NN"),
        ("EBPW-NN", "EBPW-NN"),
    ]

    results = {name: [] for name, _ in models_to_run}

    for rep in range(1, n_repetitions + 1):
        if rep % 10 == 0 or rep == 1:
            print(f"Repetition {rep} / {n_repetitions}...")

        # Generate data for this Monte Carlo draw
        X_raw, y_raw, _ = generate_simulation_data(n=n_samples, seed=42 + rep)

        # In simulation: X1, X2, X3 are parametric; X4 is non-parametric
        X_param = X_raw[:, [0, 1, 2]]
        X_nonparam = X_raw[:, 3]

        # 80/20 train/test split
        X_tr_p, X_te_p, X_tr_np, X_te_np, y_tr, y_te = train_test_split(
            X_param, X_nonparam, y_raw, test_size=0.2, random_state=42 + rep
        )

        for display_name, model_type in models_to_run:
            hybrid = HybridBPWNN(
                model_type=model_type,
                hidden_layers=1,
                hidden_nodes=50,
                epochs=epochs,
                verbose=0,
            )
            y_pred_tr, y_pred_te, num_params = hybrid.fit_and_predict(
                X_train_param=X_tr_p,
                X_test_param=X_te_p,
                X_train_nonparam=X_tr_np,
                X_test_nonparam=X_te_np,
                y_train=y_tr,
            )
            te_mse, te_r2, te_aic, te_bic = calculate_metrics(
                y_true=y_te,
                y_pred=y_pred_te,
                num_params=num_params,
                scaler_y=None,
            )
            results[display_name].append((0, 0, 0, 0, te_mse, te_r2, te_aic, te_bic))

    aggregate_cv_results(results)
    print_ablation_table(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monte Carlo Simulation Study")
    parser.add_argument("--reps", type=int, default=100, help="Number of Monte Carlo repetitions")
    parser.add_argument("--samples", type=int, default=500, help="Number of observations per repetition")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs for training residual neural network")
    args = parser.parse_args()

    run_simulation(n_repetitions=args.reps, n_samples=args.samples, epochs=args.epochs)
