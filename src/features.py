import os

import pandas as pd
import numpy as np
import plotly.express as px

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
        importance_dfs[key].to_csv(f'../checkpoints/feature_importances/{key}.csv')
    
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


def combine_feature_importances(importance_dfs):
    importance_df = pd.DataFrame({})
    
    first_key = next(iter(importance_dfs))
    
    importance_df['feature'] = importance_dfs[first_key].sort_values(by="feature")["feature"]
    
    for key in importance_dfs:
        importance_df[f'importance_{key}'] = importance_dfs[key].sort_values('feature')['importance']
        
    return importance_df

def get_feature_comparison(importance_df):
    comparison_df = importance_df
    
    comparison_df['importance_sum'] = np.zeros(len(comparison_df))
    comparison_df['rank_sum'] = np.zeros(len(comparison_df))

    for key in comparison_df:
        if 'importance_' in key and key != 'importance_sum':
            comparison_df[f'rank_{key.rsplit("_", 1)[-1]}'] = comparison_df[key].rank(ascending=False, method='min')
            
            comparison_df['importance_sum'] += comparison_df[key]
            comparison_df['rank_sum'] += comparison_df[f'rank_{key.rsplit("_", 1)[-1]}']
            
    comparison_df['rank_overall'] = comparison_df['rank_sum'].rank(method='min')
    
    return comparison_df

def plot_importance_comparison(comparison_df, num_features=15):
    df = comparison_df.sort_values('importance_sum', ascending=False).head(num_features)
    
    importance_cols = [c for c in df.columns if c.startswith("importance_") and c != 'importance_sum']
    df_imp = df.melt(
        id_vars="feature",
        value_vars=importance_cols,
        var_name="model",
        value_name="importance"
    )
    df_imp["model"] = df_imp["model"].str.replace("importance_", "")
    
    fig_imp = px.bar(
        df_imp,
        x="feature",
        y="importance",
        color="model",
        barmode="group"
    )
    
    fig_imp.show()
            
def plot_rank_comparison(comparison_df, num_features=15):
    df = comparison_df.sort_values('rank_overall').head(num_features)
    
    rank_cols = [c for c in df.columns if c.startswith("rank_") and c != 'rank_sum' and c != 'rank_overall']
    df_rank = df.melt(
        id_vars="feature",
        value_vars=rank_cols,
        var_name="model",
        value_name="rank"
    )
    df_rank["model"] = df_rank["model"].str.replace("rank_", "")
    
    fig_rank = px.bar(
        df_rank,
        x="feature",
        y="rank",
        color="model",
        barmode="group"
    )
    
    fig_rank.show()
            