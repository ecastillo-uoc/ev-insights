"""Generate split test train"""
from typing import Union, Tuple, List, Dict, Any
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
import shap


logging.basicConfig()
_logger = logging.getLogger("tfm_ev")
_logger.setLevel(logging.DEBUG)

def split_dataset(X, y, split_params:Union[Dict, None]):

    _logger.debug(f"X shape {X.shape}")
    _logger.debug(f"y shape {y.shape}")

    if split_params is None:
        split_params = {}

    test_size = split_params.get('test_size', 0.3)
    random_state = split_params.get('random_state', 42)
    shuffle = split_params.get('random_state', True)
    stratify=split_params.get('stratify', "")


    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size, 
                                                        shuffle=shuffle, stratify=stratify)
    

    return (X_train, X_test, y_train, y_test)

"""
    # Verify proportions
    print("Proporción original:")
    # print(y.value_counts(normalize=True))
    print(pd.Series(y).value_counts(normalize=True))

    print("\nProporción en train:")
    print(pd.Series(y_train).value_counts(normalize=True))

    print("\nProporción en test:")
    print(pd.Series(y_test).value_counts(normalize=True))

    return (X_train, X_test, y_train, y_test)

    for col in li_numerical_feature:
        stat, p = ks_2samp(X_train[:, li_numerical_feature.index(col)], 
                        X_test[:, li_numerical_feature.index(col)])
        if p < 0.05:
            print(f"⚠️ {col}: Distribution shift detected (p={p:.4f})")



    # Feature Stability Analysis


    # Quick model for EDA only
    rf_eda = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RANDOM_STATE)
    rf_eda.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf_eda)
    shap_values = explainer.shap_values(X_test)

    # Get feature names from X_enc (used to create X_imputed_array)
    li_feature_names = list(X_enc.columns)

    # Debug: Check where the mismatch comes from
    if False:
        print(f"X_enc shape: {X_enc.shape}")
        print(f"X_imputed_array shape: {X_imputed_array.shape}")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"Feature names: {len(li_feature_names)}")

    # Create DataFrame
    X_test_df = pd.DataFrame(X_test, columns=li_feature_names)

    # Handle shap_values format (List vs 3D Array)
    # For multiclass, shap_values can be a list of arrays (one per class) or a 3D array (samples, features, classes)
    if isinstance(shap_values, list):
        # It's a list, access class 2 directly
        print("shap_values is a list. Accessing index [2].")
        shap_values_class2 = shap_values[2]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        # It's a 3D array (samples, features, classes), slice for class 2
        print("shap_values is a 3D array. Slicing [:, :, 2].")
        shap_values_class2 = shap_values[:, :, 2]
    else:
        # Fallback
        shap_values_class2 = shap_values

    shap.summary_plot(shap_values_class2, X_test_df, max_display=10)

"""
