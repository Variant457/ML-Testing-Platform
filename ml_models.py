import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import load_dataset

AVAILABLE_MODELS = {
    'linear_regression': {
        'name': 'Linear Regression',
        'type': 'regression',
        'function': 'run_linear_regression'
    },
    'logistic_regression': {
        'name': 'Logistic Regression',
        'type': 'classification',
        'function': 'run_logistic_regression'
    }
}

def run_linear_regression(dataset_obj, target_column, parameters):
    df, error = load_dataset(dataset_obj)
    if error:
        return None, None, f"Error loading dataset: {error}"
    
    if target_column not in df.columns: #type: ignore
        return None, None, f"Target column '{target_column}' not found in dataset."
    
    try:
        X = df.drop(columns=[target_column])  #type: ignore
        y = df[target_column]  #type: ignore

        if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
            return None, None, f"All feature columns must be numeric for Linear Regression. Please preprocess the data accordingly."
    except Exception as e:
        return None, None, f"Data Preparation Error: {e}"
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression(**parameters)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'mse': mse,
        'mae': mae,
        'r2_score': r2,
        'feature_coefficients': dict(zip(X.columns, model.coef_))
    }

    return model, metrics, None

def run_logistic_regression(dataset_obj, target_column, parameters):
    df, error = load_dataset(dataset_obj)
    if error:
        return None, None, f"Error loading dataset: {error}"
    
    if target_column not in df.columns: #type: ignore
        return None, None, f"Target column '{target_column}' not found in dataset."
    
    try:
        X = df.drop(columns=[target_column])  #type: ignore
        y = df[target_column]  #type: ignore

        if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
            return None, None, f"All feature columns must be numeric for Logistic Regression. Please preprocess the data accordingly."
    except Exception as e:
        return None, None, f"Data Preparation Error: {e}"
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_params = {
        'penality': parameters.get('penalty'),
        'C': float(parameters.get('C', 1.0)),
        'solver': 'liblinear',
        'max_iter': 1000
    }

    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)\
    
    try:
        if len(model.classes_) == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovo')
    except Exception:
        roc_auc = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted'),
    recall = recall_score(y_test, y_pred, average='weighted'),
    f1 = f1_score(y_test, y_pred, average='weighted'),
    roc_auc = roc_auc

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
    }

    return model, metrics, None