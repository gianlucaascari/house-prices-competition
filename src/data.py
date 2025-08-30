import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def load_train_set():
    df = pd.read_csv('../data/preprocessed.csv')
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')
        
    return train_test_split(X, y, test_size=0.3, random_state=42), X, y
        
def load_test_data():
    X_test = pd.read_csv('../data/preprocessed_test.csv')

    for col in X_test.select_dtypes(include="object").columns:
        X_test[col] = X_test[col].astype("category")
        
    return X_test

def prepare_submission(predictions_raw):
    import json

    with open('../data/normalization_values.json', 'r') as f:
        norm_values = json.load(f)

    predictions = np.exp(predictions_raw * norm_values['std']['SalePrice'] + norm_values['mean']['SalePrice'])
    
    submission = pd.DataFrame({
        'Id': range(1461, 2920),
        'SalePrice': predictions,
    })

    submission.to_csv('../data/submission.csv', index=False)