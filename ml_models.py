import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    accuracy_score, 
    precision_score,
    recall_score, 
    f1_score, 
    roc_curve,
    precision_recall_curve,
    auc
)

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

def run_linear_regression(X, y, parameters):
    non_numeric_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    if non_numeric_cols:
        return None, None, f"Non-numeric columns found in features: {', '.join(non_numeric_cols)}. Please preprocess the data to convert these columns to numeric types."
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        model = LinearRegression(**parameters)
        model.fit(X_train, y_train)
    except Exception as e:
        return None, None, f"Model Training Error: {str(e)}"

    y_pred = model.predict(X_test)

    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2_score': r2_score(y_test, y_pred),
        'feature_coefficients': dict(zip(X.columns, model.coef_))
    }

    return model, metrics, None

def run_logistic_regression(X, y, parameters):
    non_numeric_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    if non_numeric_cols:
        return None, None, f"Non-numeric columns found in features: {', '.join(non_numeric_cols)}. Please preprocess the data to convert these columns to numeric types."
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_params = {
        'penalty': parameters.get('penalty'),
        'C': float(parameters.get('C', 1.0)),
        'solver': 'liblinear',
        'max_iter': 1000
    }

    try:
        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)
    except Exception as e:
        return None, None, f"Model Training Error: {str(e)}"

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    y_scores = y_proba[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)


    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        },
        'pr_curve': {
            'precision': precision.tolist(),
            'recall': recall.tolist()
        },
        'feature_coefficients': dict(zip(X.columns, model.coef_[0]))
    }

    return model, metrics, None