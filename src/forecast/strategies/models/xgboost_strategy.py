from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from datetime import datetime
from src.forecast.strategies.interfaces import ModelStrategy

class XGBoostStrategy(ModelStrategy):
    def __init__(self, output_key: str = 'prediction'):
        self.output_key = output_key

    def train(self, 
              df: pd.DataFrame, 
              columns: List[str], 
              target_name: str, 
              dataset_names: List[str], 
              logger: Any,
              model_name_prefix: str) -> Dict[str, Any]:
        
        output_dict = {'train': {}}
        params = {'random_state': 16, 'test_size': 0.20}

        for dataset_name in dataset_names:
            logger.info(f"{dataset_name} - XGBoost Forecast ({target_name}) - Model training")
            
            subset_df = df.loc[df['dataset_name'] == dataset_name].copy()
            
            # Prepare X (features)
            X = subset_df[columns].copy() if columns else subset_df.copy()
            
            # 1. 0/1 encoding of week days if present in columns or DF
            if 'plug_in_weekday' in X.columns:
                weekday_series = X['plug_in_weekday']
                dums = pd.get_dummies(weekday_series, prefix='plug_in_weekday')
                weekday_cols = [f'plug_in_weekday_{i}' for i in range(7)]
                dums = dums.reindex(columns=weekday_cols, fill_value=0)
                X = X.drop('plug_in_weekday', axis=1)
                X = X.join(dums)
            
            # Remove target if present
            if target_name in X.columns:
                X = X.drop(target_name, axis=1)

            # Prepare y
            # Ensure target exists
            if target_name not in subset_df.columns:
                logger.error(f"Target column {target_name} not found in dataframe")
                continue

            y = subset_df[target_name].astype(float)
            
            logger.info(f'Shape of features: {X.shape}')
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=params['test_size'], random_state=params['random_state']
            )
            
            logger.info(f'Training Features Shape: {X_train.shape}')
            
            # Train
            model = xgb.XGBRegressor(objective="reg:squarederror", random_state=params['random_state'])
            model.fit(X_train, y_train)
            
            # Predict
            preds = model.predict(X_test)
            errors = abs(preds - y_test)
            mae = np.mean(errors)
            
            # MAPE / Accuracy
            non_zero = y_test != 0
            if np.any(non_zero):
                mape = 100 * np.mean(np.abs(errors[non_zero] / y_test[non_zero]))
                accuracy = 100 - np.mean(mape)
            else:
                accuracy = 100 # Default if no usage?
            
            logger.info(f'Mean Absolute Error: {round(mae, 2)}')
            logger.info(f'Accuracy: {round(accuracy, 2)} %.')
            
            pilot_name = f"{model_name_prefix}_{dataset_name}"
            
            output_dict['train'][pilot_name] = {
                'params': params,
                'shapes': {
                    'training_features_shape': X_train.shape,
                    'training_labels_shape': y_train.shape,
                    'testing_features_shape': X_test.shape,
                    'testing_labels_shape': y_test.shape,
                },
                'model': model,
                'metrics': {
                    'mae': round(mae, 2),
                    'accuracy': round(accuracy, 2),
                },
                'artifacts': {}
            }
            
        return output_dict

    def predict(self, 
                model: Any, 
                df: pd.DataFrame, 
                columns: List[str],
                submode: str, 
                dataset_names: List[str]) -> Dict[str, Any]:
        
        output_dict = {'predict': {}}
        
        for dataset_name in dataset_names:
            if submode == 'schedule':
                subset = df.loc[df['dataset_name'] == dataset_name]
                if subset.empty:
                    continue
                
                # Use last row for prediction as per original logic
                input_row = subset.iloc[[-1]].copy()
                
                # Prepare features same as training
                # 1. Filter columns
                features = input_row[columns].copy() if columns else input_row.copy()
                
                # 2. Encode weekday
                if 'plug_in_weekday' in features.columns:
                     weekday_val = features['plug_in_weekday'].iloc[0]
                     features = features.drop('plug_in_weekday', axis=1)
                     for i in range(7):
                         features[f'plug_in_weekday_{i}'] = 1 if i == weekday_val else 0

                prediction = model.predict(features)
                val = float(prediction[0])
                
                output_dict['predict'].update({
                    self.output_key: val,
                    'value': val,
                    'date': datetime.now(),
                    'created_at': datetime.now()
                })
                
            elif submode == 'query':
                # Original logic: output_dict['predict'] = self.prediction
                pass

        return output_dict
