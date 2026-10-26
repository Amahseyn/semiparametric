"""
Hybrid Semi-Parametric Model: Parametric Estimator + Residual Neural Network.
Supports both pure parametric baselines and semi-parametric hybrid variants for ablation.
"""

import numpy as np


def build_residual_nn(input_dim=1, hidden_layers=1, hidden_nodes=100, output_dim=1):
    """
    Constructs the feedforward neural network for residual non-parametric estimation.
    Architecture:
      Input(input_dim)
      -> [Dense(hidden_nodes, relu) -> BatchNormalization] * hidden_layers
      -> Dense(output_dim)
    """
    import tensorflow as tf

    layers = [tf.keras.layers.Input(shape=(input_dim,))]
    for _ in range(hidden_layers):
        layers.append(tf.keras.layers.Dense(units=hidden_nodes, activation="relu"))
        layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.Dense(units=output_dim))

    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )
    return model


def train_residual_nn(
    X_train_nonparam,
    X_test_nonparam,
    residuals_train,
    input_dim=1,
    hidden_layers=1,
    hidden_nodes=100,
    output_dim=1,
    epochs=50,
    verbose=0,
):
    """
    Trains the residual neural network on training fold residuals.
    """
    X_train_arr = np.asarray(X_train_nonparam).reshape(len(X_train_nonparam), -1)
    X_test_arr = np.asarray(X_test_nonparam).reshape(len(X_test_nonparam), -1)
    y_res_train = np.asarray(residuals_train).reshape(-1, 1)

    model = build_residual_nn(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        hidden_nodes=hidden_nodes,
        output_dim=output_dim,
    )
    model.fit(X_train_arr, y_res_train, epochs=epochs, verbose=verbose)

    total_params = model.count_params()
    pred_res_train = model.predict(X_train_arr, verbose=0).ravel()
    pred_res_test = model.predict(X_test_arr, verbose=0).ravel()

    return pred_res_train, pred_res_test, total_params


class HybridBPWNN:
    """
    Semi-Parametric Framework supporting both pure parametric estimators
    and hybrid models with residual neural networks.
    """
    # Aliases for shorthand types that may be passed by callers
    _MODEL_ALIASES = {
        "RBPW": "RBPW-NN",
        "LBPW": "LBPW-NN",
        "EBPW": "EBPW-NN",
    }

    def __init__(
        self,
        model_type="RBPW-NN",
        hidden_layers=1,
        hidden_nodes=100,
        epochs=50,
        verbose=0,
    ):
        """
        Supported model_type options:
          Pure Parametric:
            - 'Linear Regression'
            - 'Ridge'
            - 'Lasso'
            - 'ElasticNet'
          Semi-Parametric Hybrids:
            - 'OLS-NN': Standard OLS + NN
            - 'Ridge-NN': Standard Ridge + NN
            - 'Lasso-NN': Standard Lasso + NN
            - 'Elastic-NN': Standard Elastic Net + NN
            - 'W-OLS-NN': Weighted OLS + NN (ablation: weighting alone)
            - 'RBPW-NN': Ridge BPW + NN
            - 'LBPW-NN': Lasso BPW + NN
            - 'EBPW-NN': Elastic Net BPW + NN
          Aliases (mapped automatically):
            - 'RBPW' -> 'RBPW-NN'
            - 'LBPW' -> 'LBPW-NN'
            - 'EBPW' -> 'EBPW-NN'
        """
        self.model_type = self._MODEL_ALIASES.get(model_type, model_type)
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.epochs = epochs
        self.verbose = verbose
        self.beta = None
        self.total_params = None

    def fit_and_predict(
        self,
        X_train_param,
        X_test_param,
        X_train_nonparam,
        X_test_nonparam,
        y_train,
    ):
        """
        Fits models and returns: (y_pred_train, y_pred_test, num_total_params)
        """
        from .bpw import (
            fit_bpw_estimator,
            fit_ols,
            fit_ridge,
            fit_lasso,
            fit_elastic_net,
        )

        y_train_vec = np.asarray(y_train).ravel()

        # Check if pure parametric
        is_pure_parametric = self.model_type in ["Linear Regression", "Ridge", "Lasso", "ElasticNet"]

        # Step 1: Fit parametric component
        if self.model_type in ["Linear Regression", "OLS-NN"]:
            _, beta = fit_ols(X_train_param, y_train_vec)
        elif self.model_type in ["Ridge", "Ridge-NN"]:
            _, beta, _ = fit_ridge(X_train_param, y_train_vec)
        elif self.model_type in ["Lasso", "Lasso-NN"]:
            _, beta, _ = fit_lasso(X_train_param, y_train_vec)
        elif self.model_type in ["ElasticNet", "Elastic-NN"]:
            _, beta, _, _ = fit_elastic_net(X_train_param, y_train_vec)
        elif self.model_type == "W-OLS-NN":
            _, b_ols = fit_ols(X_train_param, y_train_vec)
            beta = fit_bpw_estimator(X_train_param, y_train_vec, b_proposal=b_ols)
        elif self.model_type == "RBPW-NN":
            _, b_ridge, _ = fit_ridge(X_train_param, y_train_vec)
            beta = fit_bpw_estimator(X_train_param, y_train_vec, b_proposal=b_ridge)
        elif self.model_type == "LBPW-NN":
            _, b_lasso, _ = fit_lasso(X_train_param, y_train_vec)
            beta = fit_bpw_estimator(X_train_param, y_train_vec, b_proposal=b_lasso)
        elif self.model_type == "EBPW-NN":
            _, b_enet, _, _ = fit_elastic_net(X_train_param, y_train_vec)
            beta = fit_bpw_estimator(X_train_param, y_train_vec, b_proposal=b_enet)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.beta = beta
        y_hat_param_train = np.dot(X_train_param, beta).ravel()
        y_hat_param_test = np.dot(X_test_param, beta).ravel()

        p_param = X_train_param.shape[1] + 1  # Predictors + intercept

        # If pure parametric, return immediately
        if is_pure_parametric:
            self.total_params = p_param
            return y_hat_param_train, y_hat_param_test, p_param

        # Step 2: Fit non-parametric neural network on residuals
        residuals_train = y_train_vec - y_hat_param_train
        input_dim = 1 if np.ndim(X_train_nonparam) == 1 else X_train_nonparam.shape[1]

        pred_res_train, pred_res_test, nn_params = train_residual_nn(
            X_train_nonparam=X_train_nonparam,
            X_test_nonparam=X_test_nonparam,
            residuals_train=residuals_train,
            input_dim=input_dim,
            hidden_layers=self.hidden_layers,
            hidden_nodes=self.hidden_nodes,
            output_dim=1,
            epochs=self.epochs,
            verbose=self.verbose,
        )

        num_total_params = p_param + nn_params
        self.total_params = num_total_params

        # Step 3: Combine predictions
        y_pred_train = y_hat_param_train + pred_res_train
        y_pred_test = y_hat_param_test + pred_res_test

        return y_pred_train, y_pred_test, num_total_params
