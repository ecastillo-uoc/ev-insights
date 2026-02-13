import os
import sys
from time import time
from typing import Union, Tuple, List, Dict, Any
import re
import math

import pandas as pd

from sklearn.model_selection import GridSearchCV

import logging

logging.basicConfig()
_logger = logging.getLogger("tfm_ev.helpers")
_logger.setLevel(logging.DEBUG)


# own modules
try: 
    from .utils.logger import Colors
except Exception as ex:
    msg = str(ex)
    if msg != 'attempted relative import with no known parent package':
        print(str(ex))
    # When running directly, we are not in a module
    from utils.logger import Colors


def grid_search_model_agnostic(X_train, y_train, X_test, y_test, pipeline, 
                               param_grid:dict, scoring:str, verbose:int=0) -> Tuple[Any, Any, Any, pd.DataFrame]:
    """hyperparameters optimization grid search
    Args:
        X_train (_type_): _description_
        y_train (_type_): _description_
        X_test (_type_): _description_
        y_test (_type_): _description_
        pipeline (_type_): _description_
        param_grid (dict): _description_
        scoring (str): _description_
        verbose (int, optional): _description_. Defaults to 0.
    """

    # Check if 'oversampler' step exists in the pipeline
    if 'oversampler' in pipeline.named_steps:
        resampler_type = pipeline['oversampler'].__class__.__name__
        _logger.info(f"Resampler umbalanced classes {Colors.BOLD_BLUE}{resampler_type}{Colors.NORMAL}")
    else:
        _logger.info(f"Resampler umbalanced classes {Colors.BOLD_BLUE}None (or integrated){Colors.NORMAL}")
    
    model_type = pipeline['classifier'].__class__.__name__
    _logger.info(f"Model {Colors.BOLD_BLUE}{model_type}{Colors.NORMAL}")

    # Define a pipeline: Oversample -> Classifier
    # ADASYN/SMOTE.. will only run on the training folds during CV
    
    # Adjust param_grid to match pipeline step names (prefix with 'classifier__')
    new_param_grid = {f'classifier__{k}': v for k, v in param_grid.items()}

    # Set up GridSearchCV
    try:
        model_cv = GridSearchCV(estimator=pipeline, 
                              param_grid=new_param_grid, 
                              scoring=scoring,
                              n_jobs=-1,        # Use all available cores
                              cv=5,
                              verbose=verbose   # Verbose output to see progress =2 , no = 0
                            )   
    except ValueError as e:
        _logger.error(f"{Colors.BOLD_RED}Error al configurar GridSearchCV:{Colors.NORMAL}")
        _logger.error(f"Métrica '{scoring}' no válida o problema con param_grid")
        _logger.error(f"Error: {e}")
        raise    

    # Train the model
    _logger.info(f"Starting GridSearchCV for {model_type} model...")
    model_cv.fit(X_train, y_train)

    # Find best parameters
    best_params_raw = model_cv.best_params_
    best_params = {k.replace('classifier__', ''): v for k, v in best_params_raw.items()}
    best_scoring = model_cv.best_score_
    df_results = pd.DataFrame(model_cv.cv_results_)
    _logger.info(f"{model_type} Best parameters: {best_params} -> best cross-validated {scoring}: {best_scoring:.4f}")

    # Return variables
    best_pipeline = model_cv.best_estimator_
    best_estimator = best_pipeline.named_steps['classifier']
  

    return(best_estimator, best_params, best_scoring, df_results)