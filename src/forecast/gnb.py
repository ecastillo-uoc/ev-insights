from time import time
import numpy as np

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SMOTETomek

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

import logging

logging.basicConfig()
_logger = logging.getLogger("tfm_ev.gnb")
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

try: 
    from .helpers import grid_search_model_agnostic
except Exception as ex:
    msg = str(ex)
    if msg != 'attempted relative import with no known parent package':
        print(str(ex))
    # When running directly, we are not in a module
    from helpers import grid_search_model_agnostic


def gnb(X_train, y_train, X_test, y_test):
# Define hyperparameters for Grid search
    gnb_param_grid = { 
        'var_smoothing': np.logspace(0, -9, num=100)
    }
    scoring_gnb = 'f1_macro'  # or 'recall_macro'??


    # Define a pipeline: Oversample (SMOTETomek) -> Classifier
    pipeline_gnb = ImbPipeline([
        ('oversampler', SMOTETomek(sampling_strategy='not majority', random_state=RANDOM_STATE)),
        ('classifier', GaussianNB())
    ])

    t0 = time()
    (gnb_best_model, best_params_gnb, best_accuracy_gnb, df_results_gnb) = grid_search_model_agnostic(X_train, y_train, X_test, y_test, 
                                                                                                    pipeline_gnb, gnb_param_grid, scoring_gnb)
    t1 = time()
    execution_time_gnb = t1-t0
    aux_delta_text = f"Execution of GaussianNB in {execution_time_gnb:.2g} s"
    _logger.info(aux_delta_text)
    df_results_gnb['execution_time'] = execution_time_gnb


    return(gnb_best_model, best_params_gnb, best_accuracy_gnb, df_results_gnb)