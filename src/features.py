import os

import pandas as pd
import numpy as np

import shap

def get_shap_values_dataset(shap_values, X):
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': X.columns.values,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    return importance_df

def get_feature_importance_tree_explainer(model, X, y):
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    return get_shap_values_dataset(shap_values, X)

def get_feature_importance_kernel_explainer(model, X, y):
    model.fit(X, y)
    X_summary = shap.kmeans(X, 10)
    
    explainer = shap.KernelExplainer(model.predict, X_summary)
    shap_values = explainer.shap_values(X[:150])
    
    return get_shap_values_dataset(shap_values, X)


def compute_feature_importances(models, X, y):
    importance_dfs = {}

    for key in models:
        if models[key]['type'] == 'tree':
            importance_dfs[key] = get_feature_importance_tree_explainer(models[key]['model'], X, y)
        else:
            
            importance_dfs[key] = get_feature_importance_kernel_explainer(models[key]['model'], X, y)
            
        print(f"{key} analyzed.")
        
    return importance_dfs


def save_feature_importances(importance_dfs):
    for key in importance_dfs:
        importance_dfs[key].to_csv(f'../checkpoints/feature_importances/{key}.csv', index=False)
    
def load_feature_importances(path="../checkpoints/feature_importances"):
    importance_dfs = {}
    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            key = filename.replace(".csv", "")
            importance_dfs[key] = pd.read_csv(os.path.join(path, filename), index_col=0)
    return importance_dfs

            
def get_feature_importances(models, X, y):
    loaded_importance_dfs = load_feature_importances()
    
    # only compute values not already present
    missing_models = {}
    for key in models:
        if(key not in loaded_importance_dfs.keys()):
            missing_models[key] = models[key]
            
    print(f"Values to compute: {', '.join(missing_models.keys())}.")
            
    computed_importance_dfs = compute_feature_importances(missing_models, X, y)
    save_feature_importances(computed_importance_dfs)
    
    # only return the values for the requested models
    importance_dfs = {}
    for key in models:
        if key in loaded_importance_dfs.keys():
            importance_dfs[key] = loaded_importance_dfs[key]
        elif key in computed_importance_dfs.keys():
            importance_dfs[key] = computed_importance_dfs[key]
            
    return importance_dfs
            