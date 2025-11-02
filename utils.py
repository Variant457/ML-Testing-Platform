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

        df = None

        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        elif filename.endswith('.txt'):
            df = pd.read_csv(filepath, delimiter=r'\s+|,', engine='python')
        elif filename.endswith('.data'):
            try:
                temp_df = pd.read_csv(
                    filepath,
                    delimiter=r'\s+|,',
                    engine='python',
                )

                if not all(pd.api.types.is_numeric_dtype(temp_df[col]) for col in temp_df.columns):
                    df = temp_df
                else:
                    raise ValueError("No header detected, trying again with header=None")
            except ValueError:
                df = pd.read_csv(
                filepath,
                delimiter=r'\s+|,',
                engine='python',
                header=None
            )

            df.columns = [str(col) for col in df.columns]
        else:
            raise ValueError("Unsupported file format")
        
        df.columns = df.columns.astype(str)
        return df, None
    except pd.errors.ParserError as e:
        return None, f"Parsing error: The file appears to be corrupt or improperly formatted. Details: {str(e)}"
    except FileNotFoundError:
        return None, "File not found. Please ensure the dataset exists on the server."
    except Exception as e:
        current_app.logger.error(f"Error loading dataset: {dataset.filename}: {str(e)}")
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

def apply_transformations(df, request_form):
    try:
        df_transformed = df.copy()
                
        rename_map = {}
        col_actions = {}
        cols_to_drop = []

        for key, new_name in request_form.items():
            if key.startswith('col_rename__'):
                orig_name = key.split('__')[1]
                if new_name != orig_name and new_name:
                    if new_name in rename_map.values():
                        raise ValueError(f"Duplicate column name: '{new_name}' is used more than once.")
                    rename_map[orig_name] = new_name

        for key, value in request_form.items():
            if key.startswith('col_action__'):
                parts = key.split('__')
                if len(parts) == 3:
                    orig_name = parts[1]
                    action_type = parts[2]
                    
                    if orig_name not in col_actions:
                        col_actions[orig_name] = {}
                    
                    if action_type == 'drop' and value == 'on':
                        cols_to_drop.append(orig_name)
                    elif action_type != 'drop' and value:
                        col_actions[orig_name][action_type] = value

        df_transformed = df_transformed.rename(columns=rename_map)
        
        cols_to_remove_after_encoding = []
        cols_to_add = []

        for orig_name, actions in col_actions.items():
            col_name = rename_map.get(orig_name, orig_name)

            if orig_name in cols_to_drop:
                continue
                
            if col_name not in df_transformed.columns:
                continue

            if 'impute' in actions:
                impute_method = actions['impute']
                imputer = None
                if impute_method == 'mean':
                    if pd.api.types.is_numeric_dtype(df_transformed[col_name]):
                        imputer = SimpleImputer(strategy='mean')
                    else:
                        imputer = SimpleImputer(strategy='most_frequent')
                elif impute_method == 'median':
                    if pd.api.types.is_numeric_dtype(df_transformed[col_name]):
                        imputer = SimpleImputer(strategy='median')
                    else:
                        imputer = SimpleImputer(strategy='most_frequent')
                elif impute_method == 'mode':
                    imputer = SimpleImputer(strategy='most_frequent')
                
                if imputer:
                    df_transformed[[col_name]] = imputer.fit_transform(df_transformed[[col_name]])

            if 'scale' in actions:
                scale_method = actions['scale']
                scaler = None
                if scale_method == 'standard':
                    scaler = StandardScaler()
                elif scale_method == 'min_max':
                    scaler = MinMaxScaler()

                if scaler:
                    if pd.api.types.is_numeric_dtype(df_transformed[col_name]):
                        df_transformed[col_name] = scaler.fit_transform(df_transformed[[col_name]])
                    else:
                        pass

            if 'encode' in actions:
                encode_method = actions['encode']
                
                if encode_method == 'label':
                    le = LabelEncoder()
                    df_transformed[col_name] = le.fit_transform(df_transformed[col_name])
                elif encode_method == 'one_hot':
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    
                    df_transformed[col_name] = df_transformed[col_name].astype(str)
                    
                    new_cols_array = encoder.fit_transform(df_transformed[[col_name]])
                    new_cols_names = encoder.get_feature_names_out([col_name]) 
                    new_cols_df = pd.DataFrame(
                        new_cols_array,
                        columns=new_cols_names,
                        index=df_transformed.index
                    )
                    cols_to_add.append(new_cols_df)
                    cols_to_remove_after_encoding.append(col_name)
        if cols_to_remove_after_encoding:
            df_transformed = df_transformed.drop(columns=cols_to_remove_after_encoding)
            
        if cols_to_add:
            df_transformed = pd.concat([df_transformed] + cols_to_add, axis=1)

        if cols_to_drop:
            renamed_cols_to_drop = [rename_map.get(col, col) for col in cols_to_drop]
            renamed_cols_to_drop = [col for col in renamed_cols_to_drop if col in df_transformed.columns]
            if renamed_cols_to_drop:
                df_transformed = df_transformed.drop(columns=renamed_cols_to_drop)

        return df_transformed, None
        
    except Exception as e:
        return None, f"Error applying transformations: {str(e)}"