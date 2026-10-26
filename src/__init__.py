"""
Semi-Parametric Regression Framework Combining BPW and Neural Networks.
"""

from .bpw import (
    compute_bpw_weights,
    fit_bpw_estimator,
    fit_ols,
    fit_ridge,
    fit_lasso,
    fit_elastic_net,
)
from .metrics import calculate_metrics, print_metrics, aggregate_cv_results
from .data_loader import (
    generate_simulation_data,
    load_concrete_dataset,
    load_ct_slices_dataset,
    load_energy_dataset,
    prepare_pca_features,
)

# Lazy import for TensorFlow-dependent hybrid models
def get_hybrid_model():
    from .hybrid_model import HybridBPWNN, train_residual_nn
    return HybridBPWNN, train_residual_nn

__all__ = [
    "compute_bpw_weights",
    "fit_bpw_estimator",
    "fit_ols",
    "fit_ridge",
    "fit_lasso",
    "fit_elastic_net",
    "calculate_metrics",
    "print_metrics",
    "aggregate_cv_results",
    "generate_simulation_data",
    "load_concrete_dataset",
    "load_ct_slices_dataset",
    "load_energy_dataset",
    "prepare_pca_features",
    "get_hybrid_model",
]
