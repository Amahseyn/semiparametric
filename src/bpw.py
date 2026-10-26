"""
Balanced Penalized Weighted (BPW) Estimator and Parameter Proposals.
Supports both dense matrix and memory-efficient O(N) diagonal vector operations.
"""

import numpy as np
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    RidgeCV,
    Lasso,
    LassoCV,
    ElasticNet,
    ElasticNetCV,
)


def compute_bpw_weights(X, return_matrix=False):
    """
    Computes diagonal data-driven sample weights based on row-wise dispersion.
    w_i = 1 / sqrt(Var(X_i)) = 1 / S_i.

    Parameters:
    -----------
    X : np.ndarray of shape (n_samples, n_features)
    return_matrix : bool, default=False
        If True, returns an (n, n) diagonal matrix.
        If False, returns an (n, 1) vector to enable O(N) memory matrix multiplications.
    """
    n_samples = X.shape[0]
    # Row-wise standard deviation proxy
    variances = np.var(X, axis=1)
    # Avoid division by zero
    variances = np.where(variances > 1e-12, variances, 1.0)
    w_vec = (1.0 / np.sqrt(variances)).reshape(-1, 1)

    if return_matrix:
        return np.diag(w_vec.ravel())
    return w_vec


def fit_bpw_estimator(X, y, b_proposal, w=None, L=None):
    """
    Computes the Balanced Penalized Weighted (BPW) coefficient estimate:
        beta_BPW = (L + X^T W X)^(-1) * (L * b_proposal + X^T W y)

    Parameters:
    -----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,) or (n_samples, 1)
    b_proposal : np.ndarray of shape (n_features,) or (n_features, 1)
        Prior/proposal coefficient vector (from OLS, Ridge, Lasso, or Elastic Net).
    w : np.ndarray, optional
        Weight matrix (n, n) or weight vector (n, 1). If None, computed from X.
    L : np.ndarray of shape (n_features, n_features), optional
        Prior precision / regularization matrix. If None, sample covariance is used.

    Returns:
    --------
    beta_hat : np.ndarray of shape (n_features, 1)
    """
    if w is None:
        w = compute_bpw_weights(X, return_matrix=False)

    if L is None:
        L = np.cov(X, rowvar=False)
        if np.ndim(L) == 0:
            L = np.array([[L]])

    b_proposal = np.asarray(b_proposal).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)

    # Compute X^T W X and X^T W y efficiently
    if np.ndim(w) == 2 and w.shape[0] == w.shape[1]:
        # Dense W matrix
        XtW = np.dot(X.T, w)
    else:
        # Diagonal vector w (O(N) memory, fast vectorized multiplication)
        w_vec = np.asarray(w).reshape(-1, 1)
        XtW = X.T * w_vec.T  # Equivalent to X.T @ diag(w)

    XtWX = np.dot(XtW, X)
    XtWy = np.dot(XtW, y)

    inv_term = np.linalg.inv(L + XtWX)
    target_term = np.dot(L, b_proposal) + XtWy
    beta_hat = np.dot(inv_term, target_term)
    return beta_hat


def fit_ols(X_train, y_train):
    """Fits an unregularized Ordinary Least Squares (OLS) model."""
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)
    coef = model.coef_.reshape(-1, 1)
    return model, coef


def fit_ridge(X_train, y_train, alphas=None):
    """Fits Ridge regression with cross-validated penalty selection."""
    if alphas is None:
        alphas = [x / 100.0 for x in range(1, 1001)]  # 0.01 to 10.0
    cv = RidgeCV(alphas=alphas)
    cv.fit(X_train, y_train)
    best_alpha = cv.alpha_

    model = Ridge(alpha=best_alpha, fit_intercept=False)
    model.fit(X_train, y_train)
    coef = model.coef_.reshape(-1, 1)
    return model, coef, best_alpha


def fit_lasso(X_train, y_train, cv=5, n_alphas=100):
    """Fits Lasso regression with cross-validated penalty selection."""
    y_train_flat = np.asarray(y_train).ravel()
    cv_model = LassoCV(cv=cv, n_alphas=n_alphas, random_state=42)
    cv_model.fit(X_train, y_train_flat)
    best_alpha = cv_model.alpha_

    model = Lasso(alpha=best_alpha, fit_intercept=False)
    model.fit(X_train, y_train_flat)
    coef = model.coef_.reshape(-1, 1)
    return model, coef, best_alpha


def fit_elastic_net(X_train, y_train, l1_ratios=None, cv=5):
    """Fits Elastic Net regression with cross-validated penalty selection."""
    if l1_ratios is None:
        l1_ratios = [x / 1000.0 for x in range(1, 1001)]  # 0.001 to 1.0
    y_train_flat = np.asarray(y_train).ravel()
    cv_model = ElasticNetCV(l1_ratio=l1_ratios, cv=cv, random_state=42)
    cv_model.fit(X_train, y_train_flat)
    best_alpha = cv_model.alpha_
    best_l1_ratio = cv_model.l1_ratio_

    model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, fit_intercept=False)
    model.fit(X_train, y_train_flat)
    coef = model.coef_.reshape(-1, 1)
    return model, coef, best_alpha, best_l1_ratio
