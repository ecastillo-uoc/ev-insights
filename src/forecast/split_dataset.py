from typing import Union, Dict, Any, Tuple
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("split_dataset")

def split_dataset(X: pd.DataFrame, y: pd.Series, split_params: Union[Dict, None] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into train and test sets using sklearn.model_selection.train_test_split.
    
    Args:
        X: Feature dataframe
        y: Target series
        split_params: Dictionary with parameters:
            - test_size (float): Default 0.2
            - random_state (int): Default 42
            - shuffle (bool): Default True. Set to False for time-series sequential split.
            - stratify (bool/array): If True, stratifies by y. If array, stratifies by array.
                                     Note: stratify cannot be used if shuffle=False.
            
    Returns:
        X_train, X_test, y_train, y_test
    """
    if split_params is None:
        split_params = {}

    test_size = split_params.get('test_size', 0.2)
    random_state = split_params.get('random_state', 42)
    shuffle = split_params.get('shuffle', True)
    stratify_param = split_params.get('stratify', None)
    
    stratify = None
    if shuffle and stratify_param:
        if stratify_param is True:
            stratify = y
        else:
            stratify = stratify_param

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        random_state=random_state if shuffle else None, 
        test_size=test_size, 
        shuffle=shuffle, 
        stratify=stratify
    )

    logger.debug(f"Split completed: Train shape {X_train.shape}, Test shape {X_test.shape}")
    
    return X_train, X_test, y_train, y_test
