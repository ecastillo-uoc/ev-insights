"""
Optuna-based hyperparameter optimization for forecast models.

Each ``objective_*`` function implements a single Optuna trial:
suggest → build → train → evaluate → return metric.  The functions
are model-agnostic in principle — they use the strategy's
``build_model()`` method internally.
"""

import logging

import numpy as np
import optuna
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

from src.forecast.strategies import LSTMModelStrategy, TransformerModelStrategy
from src.forecast.strategies.utils_ts import smape

logger = logging.getLogger(__name__)

# Verbosity mode:
#   0 = silent
#   1 = progress bar
#   2 = one line per epoch.
VERBOSE_LEVEL = 1


def objective_transformer(trial, train_df, target_column):
    """Optuna objective for Transformer hyperparameter search (minimises SMAPE)."""
    PATIENCE = 10

    transformer_params = {
        "epochs": trial.suggest_int("epochs", 20, 120),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "look_back": trial.suggest_int("look_back", 7, 60),
        "head_size": trial.suggest_int("head_size", 32, 256, step=32),
        "num_heads": trial.suggest_int("num_heads", 2, 8),
        "ff_dim": trial.suggest_int("ff_dim", 8, 256, log=True),
        "num_transformer_blocks": trial.suggest_int("num_transformer_blocks", 1, 6),
        "mlp_units": trial.suggest_categorical(
            "mlp_units", [[32], [64], [128], [128, 64], [256, 128]]
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "mlp_dropout": trial.suggest_float("mlp_dropout", 0.0, 0.5),
    }

    # Safety constraint to avoid too-large models
    complexity = (
        transformer_params["head_size"]
        * transformer_params["num_heads"]
        * transformer_params["num_transformer_blocks"]
    )
    if complexity > 4096:
        raise optuna.TrialPruned()

    # Build sequence data
    subset_df = train_df.copy()
    y_raw = subset_df[[target_column]].values.astype(float)

    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y_raw)

    look_back = transformer_params["look_back"]
    X, y = [], []
    for i in range(len(y_scaled) - look_back):
        X.append(y_scaled[i : i + look_back, 0])
        y.append(y_scaled[i + look_back, 0])

    X = np.array(X)
    y = np.array(y)

    if len(X) < 50:
        raise optuna.TrialPruned()

    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Chronological train/validation split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    strategy = TransformerModelStrategy()
    model = strategy.build_model(
        input_shape=(X.shape[1], 1),
        head_size=transformer_params["head_size"],
        num_heads=transformer_params["num_heads"],
        ff_dim=transformer_params["ff_dim"],
        num_transformer_blocks=transformer_params["num_transformer_blocks"],
        mlp_units=transformer_params["mlp_units"],
        dropout=transformer_params["dropout"],
        mlp_dropout=transformer_params["mlp_dropout"],
        learning_rate=transformer_params["learning_rate"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=transformer_params["epochs"],
        batch_size=transformer_params["batch_size"],
        verbose=VERBOSE_LEVEL,
        callbacks=callbacks,
    )

    y_pred_val = model.predict(X_val, verbose=0).reshape(-1, 1)
    y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
    y_pred_inv = scaler.inverse_transform(y_pred_val).ravel()

    smape_val = smape(y_pred_inv, y_val_inv)
    trial.set_user_attr("transformer_params", transformer_params)
    return float(smape_val)


def objective_lstm(trial, train_df, target_column):
    """Optuna objective for LSTM hyperparameter search (minimises SMAPE)."""
    PATIENCE = 10

    lstm_params = {
        "epochs": trial.suggest_int("epochs", 20, 120),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "look_back": trial.suggest_int("look_back", 7, 60),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
    }

    subset_df = train_df.copy()
    y_raw = subset_df[[target_column]].values.astype(float)

    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y_raw)

    look_back = lstm_params["look_back"]
    X, y = [], []
    for i in range(len(y_scaled) - look_back):
        X.append(y_scaled[i : i + look_back, 0])
        y.append(y_scaled[i + look_back, 0])

    X = np.array(X)
    y = np.array(y)

    if len(X) < 50:
        raise optuna.TrialPruned()

    X = X.reshape((X.shape[0], X.shape[1], 1))

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    strategy = LSTMModelStrategy()
    model = strategy.build_model(
        input_shape=(X.shape[1], 1),
        learning_rate=lstm_params["learning_rate"],
        activation=lstm_params["activation"],
        dropout_rate=lstm_params["dropout_rate"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=lstm_params["epochs"],
        batch_size=lstm_params["batch_size"],
        verbose=VERBOSE_LEVEL,
        callbacks=callbacks,
    )

    y_pred_val = model.predict(X_val, verbose=0).reshape(-1, 1)
    y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
    y_pred_inv = scaler.inverse_transform(y_pred_val).ravel()

    smape_val = smape(y_pred_inv, y_val_inv)
    trial.set_user_attr("lstm_params", lstm_params)
    return float(smape_val)
