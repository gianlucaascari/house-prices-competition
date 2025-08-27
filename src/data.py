import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def load_train_set():
    df = pd.read_csv('../data/preprocessed.csv')
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')
        
    return train_test_split(X, y, test_size=0.3, random_state=42)
        
    