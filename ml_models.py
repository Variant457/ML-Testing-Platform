import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import load_dataset

AVAILABLE_MODELS = {
    'linear_regression': {
        'name': 'Linear Regression',
        'type': 'regression',
        'function': 'run_linear_regression'
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