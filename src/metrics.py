"""
Evaluation Metrics: MSE, R^2, AIC, and BIC.
Supports fold-level variability (mean +/- std) and automated ablation reporting.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred, num_params, scaler_y=None):
    """
    Computes performance metrics on the original (unscaled) target scale:
      - MSE: Mean Squared Error
      - R^2: Coefficient of Determination
      - AIC: n * ln(MSE) + 2 * num_params
      - BIC: n * ln(MSE) + num_params * ln(n)

    Parameters:
    -----------
    y_true : np.ndarray
    y_pred : np.ndarray
    num_params : int
        Total number of parameters (parametric terms + neural network weights/biases).
    scaler_y : StandardScaler, optional
        If provided, inverse-transforms y_true and y_pred to original scale.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    if scaler_y is not None:
        y_true_arr = scaler_y.inverse_transform(y_true_arr.reshape(-1, 1)).ravel()
        y_pred_arr = scaler_y.inverse_transform(y_pred_arr.reshape(-1, 1)).ravel()

    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    n = len(y_true_arr)

    # Standard least-squares formulation of information criteria
    aic = float(n * np.log(mse) + 2 * num_params)
    bic = float(n * np.log(mse) + num_params * np.log(n))

    return mse, r2, aic, bic


def print_metrics(y_train, y_test, y_pred_train, y_pred_test, model_name, num_params, scaler_y=None):
    """Prints and returns both train and test metrics for a given model."""
    train_mse, train_r2, train_aic, train_bic = calculate_metrics(
        y_train, y_pred_train, num_params, scaler_y
    )
    test_mse, test_r2, test_aic, test_bic = calculate_metrics(
        y_test, y_pred_test, num_params, scaler_y
    )
    print(f"[{model_name:<20}] Test MSE: {test_mse:.4f} | R²: {test_r2:.4f} | AIC: {test_aic:.2f} | BIC: {test_bic:.2f}")
    return (train_mse, train_r2, train_aic, train_bic, test_mse, test_r2, test_aic, test_bic)


def aggregate_cv_results(model_metrics):
    """
    Aggregates metrics across cross-validation folds.
    Prints both:
      1. Mean metrics table
      2. Mean +/- Std metrics table (variability analysis across folds)
    """
    summary = {}

    print("\n" + "=" * 70)
    print(f"{'Model':<22} | {'MSE (Mean)':<12} | {'R² (Mean)':<10} | {'AIC':<10} | {'BIC':<10}")
    print("=" * 70)
    for model_name, metrics_list in model_metrics.items():
        if not metrics_list:
            continue
        mses = [m[4] for m in metrics_list]
        r2s = [m[5] for m in metrics_list]
        aics = [m[6] for m in metrics_list]
        bics = [m[7] for m in metrics_list]

        mean_mse, std_mse = float(np.mean(mses)), float(np.std(mses))
        mean_r2, std_r2 = float(np.mean(r2s)), float(np.std(r2s))
        mean_aic = float(np.mean(aics))
        mean_bic = float(np.mean(bics))

        summary[model_name] = {
            "MSE_mean": mean_mse,
            "MSE_std": std_mse,
            "R2_mean": mean_r2,
            "R2_std": std_r2,
            "AIC": mean_aic,
            "BIC": mean_bic,
        }
        print(f"{model_name:<22} | {mean_mse:<12.4f} | {mean_r2:<10.4f} | {mean_aic:<10.2f} | {mean_bic:<10.2f}")
    print("=" * 70)

    # Variability Analysis Table (Mean +/- Std)
    print("\n" + "=" * 75)
    print("VARIABILITY ANALYSIS (Mean ± Std Across Folds)")
    print("=" * 75)
    print(f"{'Model':<22} | {'Test MSE (Mean ± Std)':<25} | {'Test R² (Mean ± Std)':<20}")
    print("-" * 75)
    for model_name, data in summary.items():
        mse_str = f"{data['MSE_mean']:.3f} ± {data['MSE_std']:.3f}"
        r2_str = f"{data['R2_mean']:.3f} ± {data['R2_std']:.3f}"
        print(f"{model_name:<22} | {mse_str:<25} | {r2_str:<20}")
    print("=" * 75)

    return summary


def print_ablation_table(model_metrics):
    """
    Generates and prints a structured Ablation Study Table.
    """
    ablation_mapping = {
        "Linear Regression": ("Parametric", "No (W=I)", "No (Lambda=0)"),
        "Ridge": ("Parametric", "No (W=I)", "Ridge (beta0=0)"),
        "OLS-NN": ("Neural Network", "No (W=I)", "No (Lambda=0)"),
        "Ridge-NN": ("Neural Network", "No (W=I)", "Ridge (beta0=0)"),
        "W-OLS-NN": ("Neural Network", "Yes (W=S^-1)", "No (Lambda=0)"),
        "RBPW-NN": ("Neural Network", "Yes (W=S^-1)", "Ridge (beta0=b_R)"),
        "LBPW-NN": ("Neural Network", "Yes (W=S^-1)", "Lasso (beta0=b_L)"),
        "EBPW-NN": ("Neural Network", "Yes (W=S^-1)", "Elastic (beta0=b_E)"),
    }

    print("\n" + "=" * 90)
    print("ABLATION STUDY: DECOMPOSITION OF MODEL COMPONENTS")
    print("=" * 90)
    print(f"{'Model Configuration':<20} | {'Residual NN':<15} | {'Weighting W':<14} | {'Target Shrinkage':<20} | {'Test MSE':<10}")
    print("-" * 90)
    for model_name, (nn_type, w_type, target_type) in ablation_mapping.items():
        if model_name in model_metrics and model_metrics[model_name]:
            mses = [m[4] for m in model_metrics[model_name]]
            mean_mse = np.mean(mses)
            print(f"{model_name:<20} | {nn_type:<15} | {w_type:<14} | {target_type:<20} | {mean_mse:<10.3f}")
    print("=" * 90)
