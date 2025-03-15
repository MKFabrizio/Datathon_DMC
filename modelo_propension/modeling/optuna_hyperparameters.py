import pandas as pd 
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from functools import partial
from sklearn import ensemble
import catboost as cb
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, log_loss, brier_score_loss, roc_auc_score, roc_curve, classification_report, make_scorer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve, CalibrationDisplay
from sklearn.metrics import roc_auc_score, r2_score

#Funciones de apoyo
def combined_metric(y_true, y_pred, y_pred_proba):
    """Calculate the combined metric."""

    #roc_auc = roc_auc_score(y_true, y_pred)
    recall_macro = recall_score(y_true,y_pred, average='macro') # REEMPLAZAR FUNCION DE RECALL AQUI
    #f1_score_score = f1_score(y_true,y_pred, average='macro')
    #prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=20, strategy='quantile')
    #r2 = r2_score(np.linspace(0, 1, len(prob_true)), prob_true)
    
    return recall_macro #*0.5 + f1_score_score*0.5 #0.7 * recall + 0.3 * r2

def objective(trial, X, y, combined_metric_function=combined_metric):
    """Objective function for Optuna optimization."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 100, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 99),
        'n_jobs': -1,
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    # Convert y to numpy array if it's a pandas Series
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    
    for train_idx, val_idx in cv.split(X, y):
        if isinstance(X, pd.DataFrame):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
        
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = XGBClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        score = combined_metric_function(y_val, y_pred, y_pred_proba)
        scores.append(score)
    
    return np.mean(scores)

def optimize_xgboost(X, y, n_trials=50, timeout=720, combined_metric_function=combined_metric):
    """Optimize XGBoost hyperparameters."""
    study = optuna.create_study(direction='maximize')
    study.optimize(partial(objective, X=X, y=y, combined_metric_function=combined_metric), n_trials=n_trials, timeout=timeout)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_params

def objective_rf(trial, X, y, combined_metric_function=combined_metric):
    """Objective function for Optuna optimization of Random Forest."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'class_weight': 'balanced',
        'bootstrap': True,
        'n_jobs': -1  # Use all available cores

    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    # Convert y to numpy array if it's a pandas Series
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    
    for train_idx, val_idx in cv.split(X, y):
        if isinstance(X, pd.DataFrame):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
        
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = ensemble.RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        score = combined_metric_function(y_val, y_pred, y_pred_proba)
        scores.append(score)
    
    return np.mean(scores)

def optimize_random_forest(X, y, n_trials=50, timeout=720, combined_metric_function=combined_metric):
    """Optimize Random Forest hyperparameters."""
    study = optuna.create_study(direction='maximize')
    study.optimize(partial(objective_rf, X=X, y=y, combined_metric_function=combined_metric), n_trials=n_trials, timeout=timeout)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_params

def objective_cat(trial, X, y, combined_metric_function=combined_metric):
    """Objective function for Optuna optimization."""
    params = {
        "objective": trial.suggest_categorical("objective", ["Logloss"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "l2_leaf_reg": trial.suggest_discrete_uniform('l2_leaf_reg', 1.0, 5.5, 0.5),
        "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
        #"rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "auto_class_weights": "Balanced",
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "30gb",
    }
    
    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        
    elif params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1)
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    # Convert y to numpy array if it's a pandas Series
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    
    for train_idx, val_idx in cv.split(X, y):
        if isinstance(X, pd.DataFrame):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
        
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = cb.CatBoostClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        score = combined_metric_function(y_val, y_pred, y_pred_proba)
        scores.append(score)
    
    return np.mean(scores)

def optimize_catboost(X, y, n_trials=50, timeout=720,combined_metric_function=combined_metric):
    """Optimize CatBoost hyperparameters."""
    study = optuna.create_study(direction='maximize')
    study.optimize(partial(objective_cat, X=X, y=y, combined_metric_function=combined_metric), n_trials=n_trials, timeout=timeout)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_params