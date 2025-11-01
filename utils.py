import os
import pandas as pd
import numpy as np
from flask import current_app
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder

def load_dataset(dataset):
    try:
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], dataset.filepath)
        filename = dataset.filename

        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        elif filename.endswith('.txt'):
            df = pd.read_csv(filepath, delimiter=r'\s+|,', engine='python')
        else:
            raise ValueError("Unsupported file format")
        return df, None
    except pd.errors.ParserError as e:
        return None, f"Parsing error: The file appears to be corrupt or improperly formatted. Details: {str(e)}"
    except FileNotFoundError:
        return None, "File not found. Please ensure the dataset exists on the server."
    except Exception as e:
        return None, f"An error occurred while loading the dataset: {str(e)}"
    
def get_data_summary(df):
    df_head_html = df.head().to_html(classes='data-preview-table', border=0, index=False)

    columns_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_count = int(df[col].isnull().sum())
        missing_percent = (missing_count / len(df)) * 100

        columns_info.append({
            'name': col,
            'dtype': dtype,
            'missing_count': missing_count,
            'missing_percent': f"{missing_percent:.2f}%",
            'is_numeric': 'int' in dtype or 'float' in dtype
        })

    shape = f"{df.shape[0]} rows, {df.shape[1]} columns"
    return {
        'head_html': df_head_html,
        'columns_info': columns_info,
        'shape': shape
    }

def apply_transformations(df, form_data):
    try:
        df_transformed = df.copy()

        actions = {}
        cols_to_drop = []
        for key, value in form_data.items():
            if not value:
                continue
            
            action, col_name = key.split('-', 1)
            if action == 'drop':
                cols_to_drop.append(col_name)
                continue

            if col_name not in actions:
                actions[col_name] = {}
            actions[col_name][action] = value

        if cols_to_drop:
            df_transformed = df_transformed.drop(columns=cols_to_drop)
            for col in cols_to_drop:
                actions.pop(col, None)

        cols_to_add = []
        cols_to_remove_after_encoding = []
        for col_name, col_acts in actions.items():
            if 'impute' in col_acts:
                method = col_acts['impute']

                if col_name not in df_transformed.columns:
                    continue

                if method == 'drop_row':
                    df_transformed = df_transformed.dropna(subset=[col_name])
                    continue

                if method in ['mean', 'median', 'mode']:
                    imputer = SimpleImputer(strategy=method)
                    df_transformed[col_name] = imputer.fit_transform(df_transformed[[col_name]])
            if 'scale' in col_acts:
                scale_method = col_acts['scale']

                if col_name not in df_transformed.columns:
                    continue

                if not pd.api.types.is_numeric_dtype(df_transformed[col_name]):
                    return None, f"Cannot apply scaling to non-numeric column '{col_name}'."

                scaler = None
                if scale_method == 'standardize':
                    scaler = StandardScaler()
                elif scale_method == 'normalize':
                    scaler = MinMaxScaler()

                if scaler:
                    df_transformed[col_name] = scaler.fit_transform(df_transformed[[col_name]])
            if 'encode' in col_acts:
                encode_method = col_acts['encode']

                if col_name not in df_transformed.columns:
                    continue

                if encode_method == 'label':
                    le = LabelEncoder()
                    df_transformed[col_name] = le.fit_transform(df_transformed[col_name])
                elif encode_method == 'one_hot':
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    new_cols_array = encoder.fit_transform(df_transformed[[col_name]])
                    new_cols_names = encoder.get_feature_names_out([col_name])
                    new_cols_df = pd.DataFrame(
                        new_cols_array,
                        columns=new_cols_names,
                        index=df_transformed.index
                    )

                    cols_to_add.append(new_cols_df)
                    cols_to_remove_after_encoding.append(col_name)
        
        if cols_to_add:
            df_transformed = pd.concat([df_transformed] + cols_to_add, axis=1)
            df_transformed = df_transformed.drop(columns=cols_to_remove_after_encoding)

        return df_transformed, None
    except Exception as e:
        return None, f"An error occurred during preprocessing: {str(e)}"