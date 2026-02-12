import lightgbm as lgb
import numpy as np
import logging
from src.forecast.strategies.interfaces import ModelStrategy
from src.forecast.strategies.ts_utils import smape
from src.forecast.forecast import Forecast

class LightGBMModelStrategy(ModelStrategy):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def train(self, df, feature_columns, target_column, dataset_names, model_name_prefix):
        output_dict = {'train': {}}
        
        # Default params from original code
        params = {'random_state': 16, 'test_size': 0.20}
        
        for dataset_name in dataset_names:
            subset_df = df.loc[df['dataset_name'] == dataset_name]
            
            X = subset_df[feature_columns]
            y = subset_df[target_column]
            
            # Simple train/test split from original code
            n_rows = len(subset_df)
            train_size = int(n_rows * 0.9)
            
            X_train, X_test = X.iloc[:train_size, :], X.iloc[train_size:, :]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            lgb_params = {
                'num_leaves': 10,
                'learning_rate': 0.02,
                'max_depth': 5,
                'verbose': 0,
                'early_stopping_rounds': 200,
                'nthread': -1
            }
            
            lgbtrain = lgb.Dataset(data=X_train, label=y_train, feature_name=feature_columns)
            lgbtest = lgb.Dataset(data=X_test, label=y_test, reference=lgbtrain, feature_name=feature_columns)
            
            lgbm_m = lgb.train(
                lgb_params,
                lgbtrain,
                valid_sets=[lgbtrain, lgbtest],
                callbacks=[lgb.early_stopping(lgb_params['early_stopping_rounds'])]
            )
            
            y_pred_test = lgbm_m.predict(X_test, num_iteration=lgbm_m.best_iteration)
            
            errors = abs(y_pred_test - y_test)
            
            # Calculate metrics
            # Avoid division by zero
            non_zero_indices = y_test != 0
            filtered_errors = errors[non_zero_indices]
            filtered_test_labels = y_test[non_zero_indices]
            
            if len(filtered_test_labels) > 0:
                mape = 100 * np.mean(np.abs(filtered_errors / filtered_test_labels))
            else:
                mape = 0
                
            accuracy = 100 - np.mean(mape)
            
            # Replicating original code's use of expm1, though strictly checks data transform
            smape_val = smape(np.expm1(y_pred_test), np.expm1(y_test)) 
            
            model_name = Forecast.get_model_name(prefix=model_name_prefix, pilot=dataset_name)
            
            output_dict['train'].update({
                model_name: {
                    'params': params,
                    'model': lgbm_m,
                    'metrics': {
                        'smape': round(smape_val, 2),
                        'mape': round(mape, 2),
                        'accuracy': round(accuracy, 2)
                    }
                }
            })
            
        return output_dict

    def predict(self, df, feature_columns, target_column, model_objects, context_date):
        # Prediction logic would go here
        return {'predict': {}}
