# Import Libraries
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

# Enable and Confirm GPU Support
# XGboost must run a "dummy test" to confirm
try:
    xgb.train(
        {"device": "cuda"},
        xgb.DMatrix([[0]], label=[0]),
        num_boost_round=1
    )
    DEVICE = "cuda"
    print("GPU confirmed. XGBoost is using CUDA.")

except Exception as gpu_error:
    try:
        xgb.train(
            {"device": "cpu"},
            xgb.DMatrix([[0]], label=[0]),
            num_boost_round=1
        )
        DEVICE = "cpu"
        print(
            f"GPU support unavailable. [Reason: {gpu_error}]\n"
            f"Model has successfully defaulted to CPU usage instead."
        )

    except Exception as cpu_error:
        DEVICE = None
        print(
            f"GPU and CPU usage are both unavailable.\n"
            f"  GPU Reason: {gpu_error}\n"
            f"  CPU Reason: {cpu_error}\n\n"
            f"Troubleshooting suggestions:\n"
            f"  1. Verify XGBoost is installed correctly  : pip install xgboost\n"
            f"  2. Verify your CUDA drivers are up to date : https://developer.nvidia.com/cuda-downloads\n"
            f"  3. Confirm your Python environment is not corrupted\n"
            f"  4. Try reinstalling XGBoost                : pip install --force-reinstall xgboost"
        )


# Define file path & Load Dataset
DATA_DIR   = "../datasets"
TRAIN_FILE = "balanced_train.txt"

train_path = os.path.join(DATA_DIR, TRAIN_FILE)
df = pd.read_csv(train_path)

# XGBOOST CONFIGURATIONS
config = {
    
    #GLOBAL & GENERAL CONFIGURATIONS
    "verbosity"            : 1,                # 0 (silent), 1 (warning), 2 (info), 3 (debug)
    # "use_rmm"               : False,            # DEFAULT: FALSE -- RMM XGBoost not required or installed (Only relevant for production pipelines)
    # "use_cuda_async_pool"   : False,            # DEFAULT: FALSE -- Uses CUDA's built-in memory pool for faster GPU memory allocation.
    # "nthread" - Only Applicable for overriding multi-threading CPU behavior. [Default is to auto-detect and use all cores]
    "booster"              : "gbtree",         # DEFAULT: gbtree --  Options: "gbtree" (Standard Gradient Boosting), "gblinear" (Linear Models), "dart" (Dropout Trees)
    "device"               : DEVICE,           # Options: "cuda", "cpu" - Specified in cell 2
    "validate_parameters"  : True,          # Use with Python. XGBoost validates input parameters and warns if any are unused or unrecognized.
    "disable_default_eval_metric" : False,    # XGBoost supports variety of "eval_metric" options with defaults and customizability.

    # TREE BOOSTER PARAMETERS
    "eta"                  : 0.3,              # DEFAULT: 0.3 - Alias for "learning_rate". Controls step size shrinkage to prevent overfitting.
    "gamma"                : 0,                # DEFAULT: 0 - Minimum loss reduction required to make a split. Higher values = more conservative models.
    "max_depth"            : 6,                # DEFAULT: 6 - Maximum depth of a tree. Increasing may capture more complex patterns BUT may overfit.
    "min_child_weight"     : 1,                # DEFAULT: 1 - [Options: 0,∞] Minimum sum of instance weight (hessian) needed in a child. Higher values prevent overfitting.
                                               # Relevant for U2R and R2L (Low # might cause overfitting)
    "max_delta_step"       : 0,                # DEFAULT: 0 - Maximum delta step allowed. 0 = no constraints. May help with logistic regressionin imbalanced datasets.
    "subsample"           : 0.5,                # DEFAULT: 1 - [Range: 0,1]Randomly subsample dataset prior to growing tree. May prevent overfitting. Occurs once per boosting iteration.
    "sampling_method"      : "uniform",        # DEFAULT: uniform - Options: "uniform" (Random sampling), "gradient_based" (Gradient-based sampling)
    # ALL parameters in the "column" sampling family remain at their default value 1 to disable random column sampling.
    # This is to ensure no uncontrolled variables are introduced in the feature selection process.
    "lambda"               : 1,                # DEFAULT: 1 - [Options: 0,∞] (Alias: reg_lambda) L2 regularization term on weights. Increase = more conservative models.
    "alpha"                : 0,                # DEFAULT: 0 - [Options: 0,∞] (Alias: reg_alpha) L1 regularization term on weights. Increase = more conservative models.
    "tree_method"         : "hist",            # DEFAULT: auto - Options: "auto" (same as "hist"), "exact" (Exact greedy algorithm), "approx" (Approximate algorithm using quantile sketching), "hist" (Histogram-based algorithm)
    # "scale_pos_weight"  : 1,                   # DEFAULT: 1 - Not applicable for multiclass classifcation. (Only relevant for binary classification)
    # "updater"         #Extremely advanced parameter set automatically. Not recommended for manual tuning. See documentation for details.
    # "refresh_leaf"    # Related to "updater" parameter. Not recommended for manual tuning. See documentation for details.
    "process_type"        : "default",         # DEFAULT: default - Options: "default" (Standard training), "update" (Continue training from existing model)
    # "grow_policy"        : "depthwise",       # DEFAULT: depthwise - Options: "depthwise" (Split at nodes with highest depth first. More controlled, less overfitting)), "lossguide" (Split at nodes with highest loss reduction first, prone to overfitting)
    # "max_leaves"         : 0,                # DEFAULT: 0 - Maximum number of leaves in a tree. Only relevant when grow_policy=lossguide. Higher values may lead to overfitting.
    # "max_bin"           : 256,              # DEFAULT: 256 - Number of bins for histogram-based algorithms. Higher values may lead to better accuracy (esp for U2R & R2L) but slower training.
    # "num_parallel_tree" : 1,                # DEFAULT: 1 - Number of trees to grow per round (Supports Boosted Random Forest). Keep at 1 to prevent affecting feature importance scores.
    # "monotonic_constraints" : None,             # DEFAULT: None - Ignore for Feature Selection purposes.
    # "interaction_constraints" : None,            # DEFAULT: None - Ignore for Feature Selection purposes.
    # "max_cached_hist_node" : 65536,            # DEFAULT: 65536 - Maximum number of nodes to cache for histogram-based algorithms. Increase = speed up training + consume more memory.
    # "max_cat_to_onehot" Ignore - we are explicitely not using One-Hot Encoding
    # Skip all parameters related to Dart Booster
    # Skip all parameters related to Linear Booster
    "objective"            : "multi:softprob", # Many other options (see documentation), but "multi:softmax" is likely the most promising alternative.
    # "base_score" -- Automatically estimated by XGBoost for multi:softprob — no manual override needed
    "eval_metric" : "mlogloss", # Defaults to "mlogloss" for multi:softprob, but can be overridden with other options. See documentation.
    "seed" : 0, # RNG to ensure reproducibility of results across different runs/environments.

    # Parameters related to Tweedie Regression, Pseudo-Huber, Quantile Loss, AFT Survival Loss skipped - irrelevant for multiclass classification.
    
    # PARAMETERS NOT LISTED IN DOCUMENTATION BUT RELEVANT FOR FEATURE SELECTION
    "enable_categorical" : True, # Avoids using One-Hot Encoding and allows XGBoost to natively handle categorical features. 
    "num_class"            : 5,  # Required for multiclass classification with "multi:softprob" objective. Specifies number of classes in the dataset.
    "n_estimators"         : 100,              # DEFAULT: 100 - Number of boosting rounds (trees).
    "early_stopping_rounds": 10,        # DEFAULT: None -- Early stopping if no imporvement (eval metric) for specified rounds.
    # Prototyping
    "prototyping"          : False,
    "prototyping_frac"     : 0.2,              # fraction of data to use when prototyping
    "prototyping_stratified": True,            # ensure all classes are represented
}

