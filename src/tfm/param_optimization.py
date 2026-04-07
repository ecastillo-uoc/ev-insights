import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from datetime import timedelta

import optuna


from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

from src.forecast.forecast_implementations import TransformerModelStrategy
from src.forecast.strategies.utils_ts import smape

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Verbosity mode:
#   0 = silent 
#   1 = progress bar
#   2 = one line per epoch.
#   "auto" becomes 1 for most cases. 
VERBOSE_LEVEL = 1
use_cudnn = True

def objective_transformer(trial):

    # Increase epochs and rely on early stopping
    N_TRIAL_EPOCHS = 10 
    PATIENCE = 10


    # -------------------------------------------------------------------------
    # Hyperparameters to optimize
    # -------------------------------------------------------------------------

    #  hyperparameters with the same keys used by TransformerModelStrategy
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

    # Optional safety constraint to avoid too-large models
    complexity = (
        transformer_params["head_size"]
        * transformer_params["num_heads"]
        * transformer_params["num_transformer_blocks"]
    )
    if complexity > 4096:
        raise optuna.TrialPruned()


    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    # Build sequence data from your training series
    # Replace this with your real dataframe preparation:
    # subset_df must be one dataset_name and include target column
    
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


    # Chronological train/validation split (important for TS)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]


    # Build model using the same strategy class
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

    # -------------------------------------------------------------------------
    # Compile
    # -------------------------------------------------------------------------

    
    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True
        )
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=transformer_params["epochs"],
        batch_size=transformer_params["batch_size"],
        verbose=VERBOSE_LEVEL,
        callbacks=callbacks,
    )


    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------


    # Evaluate in original scale with SMAPE (or MAE/MSE)
    y_pred_val = model.predict(X_val, verbose=0).reshape(-1, 1)
    y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
    y_pred_inv = scaler.inverse_transform(y_pred_val).ravel()

    smape_val = smape(y_pred_inv, y_val_inv)

    # Save params for later reuse in your pipeline
    trial.set_user_attr("transformer_params", transformer_params)

    # Minimize SMAPE
    return float(smape_val)



def objective_lstm(trial):

    # Increase epochs and rely on early stopping
    N_TRIAL_EPOCHS = 10 
    PATIENCE = 10


    # -------------------------------------------------------------------------
    # Hyperparameters to optimize
    # -------------------------------------------------------------------------

    #  hyperparameters with the same keys used by LSTMModelStrategy
    lstm_params = {
        "epochs": trial.suggest_int("epochs", 20, 120),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "look_back": trial.suggest_int("look_back", 7, 60),
        
        "activation": trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
    }


    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    # Build sequence data from your training series
    # Replace this with your real dataframe preparation:
    # subset_df must be one dataset_name and include target column
    
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


    # Chronological train/validation split (important for TS)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]


    # Build model using the same strategy class
    from src.forecast.forecast_implementations import LSTMModelStrategy
    strategy = LSTMModelStrategy()
    model = strategy.build_model(
        input_shape=(X.shape[1], 1),
        learning_rate=lstm_params["learning_rate"],
        activation=lstm_params["activation"],
        dropout_rate=lstm_params["dropout_rate"],
    )

    # -------------------------------------------------------------------------
    # Compile
    # -------------------------------------------------------------------------

    
    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True
        )
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=lstm_params["epochs"],
        batch_size=lstm_params["batch_size"],
        verbose=VERBOSE_LEVEL,
        callbacks=callbacks,
    )


    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------


    # Evaluate in original scale with SMAPE (or MAE/MSE)
    y_pred_val = model.predict(X_val, verbose=0).reshape(-1, 1)
    y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
    y_pred_inv = scaler.inverse_transform(y_pred_val).ravel()

    smape_val = smape(y_pred_inv, y_val_inv)

    # Save params for later reuse in your pipeline
    trial.set_user_attr("lstm_params", lstm_params)

    # Minimize SMAPE
    return float(smape_val)