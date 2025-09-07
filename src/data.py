import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def load_train_data():
    df_train = pd.read_csv('../data/preprocessed/dfn_train.csv')
    df_val = pd.read_csv('../data/preprocessed/dfn_val.csv')
    
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    
    X_val = df_val.iloc[:, :-1]
    y_val = df_val.iloc[:, -1]
    
    X = pd.concat([X_train, X_val], axis=0)
    y = pd.concat([y_train, y_val], axis=0)

    full_df = pd.concat([X, y], axis=1)
        
    return X_train, X_val, y_train, y_val, full_df
        
def load_test_data():
    df = pd.read_csv('../data/preprocessed/dfn.csv')
    df_test = pd.read_csv('../data/preprocessed/dfn_test.csv')
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_test = df_test
        
    return X, y, X_test

def prepare_submission(predictions_raw):
    import json

    with open('../data/test_normalization_values.json', 'r') as f:
        norm_values = json.load(f)

    predictions = np.exp(predictions_raw * norm_values['std']['SalePrice'] + norm_values['mean']['SalePrice'])
    
    submission = pd.DataFrame({
        'Id': range(1461, 2920),
        'SalePrice': predictions,
    })

    submission.to_csv('../data/submission.csv', index=False)