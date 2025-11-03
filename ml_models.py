import traceback
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neural_network import MLPClassifier, MLPRegressor
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
    },
    'dnn_classifier_pytorch': {
        'name': 'DNN Classifier (PyTorch)',
        'type': 'classification',
        'function': 'run_dnn_classifier_pytorch'
    },
    'dnn_regressor_pytorch': {
        'name': 'DNN Regressor (PyTorch)',
        'type': 'regression',
        'function': 'run_dnn_regressor_pytorch'
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

def parse_hidden_layers(layers_str):
    if not layers_str:
        return (100,)
    try:
        layers = tuple(int(x.strip()) for x in layers_str.split(',') if x.strip())
        if not layers:
            return (100,)
        return layers
    except ValueError:
        raise ValueError(f"Invalid hidden layer format. Please use comma-separated integers, e.g., '100, 50'. You entered: '{layers_str}'")
    
class DynamicMLP(nn.Module):
    def __init__(self, input_features, output_classes, hidden_layers_tuple, activation_str):
        super(DynamicMLP, self).__init__()
        
        activation_map = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'logistic': nn.Sigmoid}
        if activation_str not in activation_map:
            raise ValueError("Unsupported activation function")
            
        act_fn = activation_map[activation_str]
        
        layers = []
        last_dim = input_features
        
        for layer_size in hidden_layers_tuple:
            layers.append(nn.Linear(last_dim, layer_size))
            layers.append(act_fn())
            last_dim = layer_size
            
        layers.append(nn.Linear(last_dim, output_classes))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def run_dnn_classifier_pytorch(X, y, parameters):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long) 
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
        
        batch_size = 64
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        hidden_layers = parse_hidden_layers(parameters.get('hidden_layer_sizes_str', '100'))
        activation = parameters.get('activation', 'relu')
        input_features = X.shape[1]
        output_classes = len(y.unique()) 
        
        model = DynamicMLP(input_features, output_classes, hidden_layers, activation)
        
        epochs = int(parameters.get('max_iter', 200))
        lr = float(parameters.get('learning_rate_init', 0.001))
        alpha = float(parameters.get('alpha', 0.0001))
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)
        
        model.train() 
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        model.eval() 
        all_preds = []
        all_true = []
        all_probas = []

        with torch.no_grad(): 
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                
                probas = nn.functional.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.numpy())
                all_true.extend(y_batch.numpy())
                all_probas.extend(probas.numpy())
                
        all_probas = np.array(all_probas)
        
        y_scores = all_probas[:, 1]
        fpr, tpr, _ = roc_curve(all_true, y_scores)
        roc_auc = auc(fpr, tpr)
            
        metrics = {
            'accuracy': accuracy_score(all_true, all_preds),
            'precision': precision_score(all_true, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_true, all_preds, average='weighted', zero_division=0),
            'f1_score': f1_score(all_true, all_preds, average='weighted', zero_division=0),
            'roc_auc': roc_auc,
            'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
        }
        return None, metrics, None
        
    except ValueError as ve:
        return None, None, str(ve)
    except Exception as e:
        traceback.print_exc()
        return None, None, f"PyTorch Model failed: {str(e)}"