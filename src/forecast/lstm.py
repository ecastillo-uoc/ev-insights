import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from src.forecast.forecast import Forecast

class LSTMForecast(Forecast):
    def __init__(self, id, name, algo, info, actor, actor_id, date, enabled, full_custom_mode, mode, submode, models_dir, model_name,
                 show_images, save_images, save_results, input_interface, output_interface, mlflow_interface, output_dir, data_selection,
                 custom_params):
        super().__init__(id=id, name=name, algo=algo, info=info, actor=actor, actor_id=actor_id, date=date, enabled=enabled,
                         full_custom_mode=full_custom_mode, mode=mode, submode=submode, models_dir=models_dir, model_name=model_name,
                         show_images=show_images, save_images=save_images, save_results=save_results, input_interface=input_interface,
                         output_interface=output_interface, mlflow_interface=mlflow_interface, output_dir=output_dir,
                         data_selection=data_selection, custom_params=custom_params)
        
        # Hyperparameters according to requested configuration with default values
        lstm_params = custom_params.get('lstm_params', {}) if custom_params else {}
        self.epochs = lstm_params.get('epochs', 100)
        self.batch_size = lstm_params.get('batch_size', 32)
        self.learning_rate = lstm_params.get('learning_rate', 0.001)
        self.activation = lstm_params.get('activation', 'relu')
        self.dropout_rate = lstm_params.get('dropout_rate', 0.2)
        self.prediction_days = lstm_params.get('prediction_days', 30) # Default to 30 days horizon
        self.look_back = lstm_params.get('look_back', 30) # Number of previous days used to predict the next
        
        # Scaler
        self.scaler = MinMaxScaler()
        self.model = None

    def build_model(self, input_shape):
        """
        Builds the LSTM model using the defined hyperparameters.
        """
        model = Sequential()
        
        # First LSTM layer with Dropout
        model.add(LSTM(units=50, activation=self.activation, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(self.dropout_rate))
        
        # Second LSTM layer with Dropout
        model.add(LSTM(units=50, activation=self.activation))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Optimizer
        optimizer = Adam(learning_rate=self.learning_rate)
        
        # Error metrics: MAE, MSE
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        return model

    def prepare_data(self, df, feature_col, target_col, time_steps=1):
        """
        Prepares the dataset, applies MinMaxScaler, and generates time steps for LSTM.
        """
        data = df[[feature_col, target_col]].values
        
        # Scale the data using MinMaxScaler
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - time_steps):
            X.append(scaled_data[i:(i + time_steps), 0])
            y.append(scaled_data[i + time_steps, 1])
            
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y

    def train(self, X_train, y_train):
        """
        Trains the LSTM model.
        """
        input_shape = (X_train.shape[1], 1)
        self.model = self.build_model(input_shape)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )
        return history

    def predict(self, initial_sequence):
        """
        Predicts iteratively into the future for 'prediction_days' using the trained LSTM model.
        
        Args:
            initial_sequence: The last 'look_back' known days to start the prediction from.
            
        Returns:
            predictions: The iterative predictions for 'prediction_days'.
        """
        if self.model is None:
            raise ValueError("The model must be trained before predicting.")
            
        predictions = []
        current_seq = initial_sequence.copy()
        
        # Iteratively predict 'prediction_days' into the future
        for _ in range(self.prediction_days):
            # Assumes input shape [1, time_steps, features]
            pred = self.model.predict(current_seq[np.newaxis, :, :], verbose=0)
            predictions.append(pred[0, 0])
            
            # Slide the window: remove oldest, append newest prediction
            current_seq = np.roll(current_seq, -1, axis=0)
            current_seq[-1, 0] = pred[0, 0]
            
        return np.array(predictions)


    def run(self):
        """
        Main execution workflow.
        """
        self.logger.info("Running LSTM Forecast...")
        
        # Place standard pipeline execution here depending on custom_params 
        # normally: feature_engineering() -> train() -> predict()
        pass
