from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

MODELS = {
    "xgb": {
        "name": "XGBoost Regressor",
        "model": XGBRegressor(),
        "type": "tree",
    },
    # "ctb": {
    #     "name": "CatBoost Regressor",
    #     "model": CatBoostRegressor(verbose=False),
    #     "type": "tree",
    # },
    # "lgb": {
    #     "name": "LightGBM Regressor",
    #     "model": LGBMRegressor(verbose=-1),
    #     "type": "tree",
    # },
    "svr": {
        "name": "Support Vector Regressor",
        "model": SVR(),
        "type": "other",
    },
    "mlp": {
        "name": "MLP Regressor",
        "model": MLPRegressor(),
        "type": "other",
    },
    "knn": {
        "name": "KNN Regressor",
        "model": KNeighborsRegressor(),
        "type": "other",
    },
}
