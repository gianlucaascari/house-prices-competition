from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import cross_val_score

MODELS = {
    "xgb": {
        "name": "XGBoost Regressor",
        "model": XGBRegressor,
        "type": "tree",
    },
    "ctb": {
        "name": "CatBoost Regressor",
        "model": CatBoostRegressor,
        "type": "tree",
    },
    "lgb": {
        "name": "LightGBM Regressor",
        "model": LGBMRegressor,
        "type": "tree",
    },
    "svr": {
        "name": "Support Vector Regressor",
        "model": SVR,
        "type": "other",
    },
    "mlp": {
        "name": "MLP Regressor",
        "model": MLPRegressor,
        "type": "other",
    },
    "knn": {
        "name": "KNN Regressor",
        "model": KNeighborsRegressor,
        "type": "other",
    },
}

def evaluate_models(models, X, y):
    scores = {}
    
    for key in models:
        model = models[key]['model']
        scores[key] = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        
    return scores
        
