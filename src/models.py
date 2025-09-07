import copy
import json

import pandas as pd
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Lasso, Ridge, ElasticNet

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import root_mean_squared_error

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor

XGB = {
    "name": "XGBoost Regressor",
    "model": XGBRegressor,
    "type": "tree",
    'params_search_space': {
        'n_estimators': {
            'low_limit': 100,
            'top_limit': 2000,
            'type': 'int',
            'log': True,
        },
        'learning_rate':  {
            'low_limit': 0.005,
            'top_limit': 0.3,
            'type': 'float',
            'log': True,
        },
        'max_depth': {
            'low_limit': 2,
            'top_limit': 16,
            'type': 'int',
            'log': True,
        },
    },
    'base_params': {},
}

SVR = {
    "name": "Support Vector Regressor",
    "model": SVR,
    "type": "other",
    'params_search_space': {
        'C': {
            'low_limit': 0.1,
            'top_limit': 100,
            'type': 'float',
            'log': True,
        },
        'gamma':  {
            'low_limit': 0.001,
            'top_limit': 1,
            'type': 'float',
            'log': True,
        },
        'epsilon': {
            'low_limit': 0.001,
            'top_limit': 1,
            'type': 'float',
            'log': True,
        },
    },
    'base_params': {},
}

MLP = {
    "name": "MLP Regressor",
    "model": MLPRegressor,
    "type": "other",
    'params_search_space': {
        'learning_rate_init': {
            'low_limit': 1e-4,
            'top_limit': 1,
            'type': 'float',
            'log': True,
        },
        'alpha':  {
            'low_limit': 0.001,
            'top_limit': 1,
            'type': 'float',
            'log': True,
        },
        "hidden_layer_sizes": {
            "type": "layers",
            "n_layers_min": 1,
            "n_layers_max": 1,
            "layers_log": False,
            "n_neurons_low": 4,
            "n_neurons_high": 256,
            "log": True,
        },
    },
    'base_params': {
        'max_iter': 2000,
    }
}

KNN = {
    "name": "KNN Regressor",
    "model": KNeighborsRegressor,
    "type": "other",
    'params_search_space': {
        'n_neighbors': {
            'low_limit': 1,
            'top_limit': 500,
            'type': 'int',
            'log': True,
        },
        'weights':  {
            'choices': ['uniform', 'distance'],
            'type': 'categorical',
        },
    },
    'base_params': {}
}

CTB = {
    "name": "CatBoost Regressor",
    "model": CatBoostRegressor,
    "type": "tree",
    'params_search_space': {
        'depth': {
            'low_limit': 2,
            'top_limit': 10,
            'type': 'int',
        },
        'learning_rate':  {
            'low_limit': 1e-3,
            'top_limit': 0.3,
            'type': 'float',
            'log': True,
        },
        'l2_leaf_reg':  {
            'low_limit': 1,
            'top_limit': 10,
            'type': 'float',
            'log': True,
        },
        'iterations':  {
            'low_limit': 200,
            'top_limit': 2000,
            'type': 'int',
            'log': True,
        },
    },
    'base_params': {
        'loss_function': 'RMSE',
        'verbose': False,
    }
}

LGB = {
    "name": "LightGBM Regressor",
    "model": LGBMRegressor,
    "type": "tree",
    'params_search_space': {
        'num_leaves': {
            'low_limit': 2,
            'top_limit': 256,
            'type': 'int',
            'log': True,
        },
        'learning_rate':  {
            'low_limit': 1e-3,
            'top_limit': 0.3,
            'type': 'float',
            'log': True,
        },
        'min_data_in_leaf':  {
            'low_limit': 10,
            'top_limit': 4e3,
            'type': 'int',
            'log': True,
        },
        'max_depth':  {
            'low_limit': 2,
            'top_limit': 32,
            'type': 'int',
            'log': True,
        },
    },
    'base_params': {
        'verbose': -1,
    }
}

LASSO = {
    "name": "Lasso",
    "model": Lasso,
    "type": "other",
    'params_search_space': {
        'alpha': {
            'low_limit': 1e-3,
            'top_limit': 1e3,
            'type': 'float',
            'log': True,
        },
    },
    'base_params': {}
}

RIDGE = {
    "name": "Ridge",
    "model": Ridge,
    "type": "other",
    'params_search_space': {
        'alpha': {
            'low_limit': 1e-3,
            'top_limit': 1e3,
            'type': 'float',
            'log': True,
        },
    },
    'base_params': {}
}

ELNET = {
    "name": "Elastic Net",
    "model": ElasticNet,
    "type": "other",
    'params_search_space': {
        'alpha': {
            'low_limit': 1e-3,
            'top_limit': 1e3,
            'type': 'float',
            'log': True,
        },
        'l1_ratio': {
            'low_limit': 0.0,
            'top_limit': 1.0,
            'type': 'float',
            'log': False,
        },
    },
    'base_params': {}
}



MODELS = {
    "xgb": XGB,
    "ctb": CTB,
    "lgb": LGB,
    "svr": SVR,
    "mlp": MLP,
    "knn": KNN,
}

META_MODELS = {
    'lasso': LASSO,
    'ridge': RIDGE,
    'elNet': ELNET,
    'svr': SVR,
    # 'mlp': MLP,
}

def make_objective_function(model_entity, search_space, X, y, cv, scoring="neg_mean_squared_error"):
    """
    Returns an Optuna objective function for a given model and parameter search space.
    """
    def objective(trial):
        params = model_entity['base_params']
        layers = None

        for pname, spec in search_space.items():
            if spec["type"] == "layers":
                num_layers = trial.suggest_int("num_layers", spec['n_layers_min'], spec['n_layers_max'], log=spec.get("layers_log", False))
                layers = []
                for i in range(num_layers):
                    n_neurons = trial.suggest_int(
                        f"n_neurons_{i}",
                        spec["n_neurons_low"],
                        spec["n_neurons_high"],
                        log=spec.get("log", False),
                    )
                    layers.append(n_neurons)
            elif spec["type"] == "int":
                params[pname] = trial.suggest_int(
                    pname, spec["low_limit"], spec["top_limit"], log=spec.get("log", False)
                )
            elif spec["type"] == "float":
                params[pname] = trial.suggest_float(
                    pname, spec["low_limit"], spec["top_limit"], log=spec.get("log", False)
                )
            elif spec["type"] == "categorical":
                params[pname] = trial.suggest_categorical(pname, spec["choices"])
            else:
                raise ValueError(f"Unsupported param type {spec['type']}")

        if layers is not None:
            params["hidden_layer_sizes"] = tuple(layers)

        model = model_entity['model'](**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return np.sqrt(-scores.mean())

    return objective

def build_model_params(model_entity, best_params):
    params = model_entity['base_params']
    layers = None

    for pname, spec in model_entity['params_search_space'].items():
        if spec["type"] in ["int", "float", "categorical"]:
            # Se il parametro è presente, prendi il valore da best_params
            if pname in best_params:
                params[pname] = best_params[pname]
        elif spec["type"] == "layers":
            num_layers = best_params["num_layers"]
            layers = []
            for i in range(num_layers):
                layers.append(best_params[f"n_neurons_{i}"])
        else:
            raise ValueError(f"Unsupported param type {spec['type']}")

    if layers is not None:
        params["hidden_layer_sizes"] = tuple(layers)

    return params

def optimize_and_evaluate_model(model_entity, X_train, y_train, X_val, y_val, model_selection = {}, features_by_importance = []):
    objective = make_objective_function(model_entity, model_entity['params_search_space'], X_train, y_train, 5)
        
    study = optuna.create_study(direction='minimize')
    
    study.optimize(objective, n_trials=100)
    
    best_params = build_model_params(model_entity, study.best_params)
    
    model = model_entity['model'](**best_params)
    cv_score = np.sqrt(-cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    val_score = root_mean_squared_error(y_val, y_pred)
    
    return best_params, pd.DataFrame(cv_score), val_score, study
    

def evaluate_best_feature_number(model_entity, ranked_features, numbers, X_train, y_train, X_val, y_val):
    params = {}
    cv_scores = {}
    val_scores = {}
    studies = {}
    
    for number in numbers:
        print(f'processsing number {number}')
        features = ranked_features[:number]
        param, cv_score, val_score, study = optimize_and_evaluate_model(model_entity, X_train[features], y_train, X_val[features], y_val)
        
        print(f'cv_score: {cv_score.mean()[0]}, val_score: {val_score}, param: {param}')
        
        params[str(number)] = copy.deepcopy(param)
        cv_scores[str(number)] = cv_score
        val_scores[str(number)] = val_score
        studies[str(number)] = study
        
    df_scores = pd.concat(cv_scores, axis=1)
    df_scores.columns = df_scores.columns.droplevel(1)
        
    return params, df_scores, val_scores, studies

def save_models_values(params, cv_scores, val_scores):
    with open('../checkpoints/model_optimization/params.json', 'w', encoding='utf-8') as f:
        json.dump(params, f)
    
    cv_scores_serializable = {k: df.to_dict(orient="records") for k, df in cv_scores.items()}
    with open('../checkpoints/model_optimization/cv_scores.json', 'w', encoding='utf-8') as f:
        json.dump(cv_scores_serializable, f, indent=2)
        
    with open('../checkpoints/model_optimization/val_scores.json', 'w', encoding='utf-8') as f:
        json.dump(val_scores, f)

def load_models_values():
    with open('../checkpoints/model_optimization/params.json', 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    with open('../checkpoints/model_optimization/cv_scores.json', 'r', encoding='utf-8') as f:
        cv_scores_serializable = json.load(f)
    cv_scores = {k: pd.DataFrame(v) for k, v in cv_scores_serializable.items()}
    
    with open('../checkpoints/model_optimization/val_scores.json', 'r', encoding='utf-8') as f:
        val_scores = json.load(f)
    
    return params, cv_scores, val_scores


def plot_model_scores(cv_scores, val_scores, feature_numbers, items_per_row=2):
    value_vars = [str(n) for n in feature_numbers]

    n_models = len(cv_scores)
    n_rows = (n_models + items_per_row - 1) // items_per_row

    fig = sp.make_subplots(
        rows=n_rows,
        cols=items_per_row,
        subplot_titles=list(cv_scores.keys())
    )

    for i, model_name in enumerate(cv_scores.keys()):
        df = cv_scores[model_name]

        # melt results for CV
        df_res = df.reset_index().melt(
            id_vars='index',
            value_vars=value_vars,
            var_name='number of features',
            value_name='score'
        )

        # compute mean/std across folds
        df_stats = df_res.groupby('number of features')['score'].agg(['mean', 'std']).reset_index()

        # validation scores in un DataFrame per comodità
        df_val = pd.DataFrame({
            'number of features': value_vars,
            'val_score': [val_scores[model_name][n] for n in value_vars]
        })

        row = i // items_per_row + 1
        col = i % items_per_row + 1

        # add CV mean+std bar
        fig.add_trace(
            go.Bar(
                x=df_stats['number of features'],
                y=df_stats['mean'],
                error_y=dict(type='data', array=df_stats['std']),
                name=f"{model_name} CV",
            ),
            row=row, col=col
        )

        # add Validation bar
        fig.add_trace(
            go.Bar(
                x=df_val['number of features'],
                y=df_val['val_score'],
                name=f"{model_name} Validation",
            ),
            row=row, col=col
        )

    fig.update_layout(
        barmode='group',  # per metterle affiancate
        height=400 * n_rows,
        width=500 * items_per_row,
        showlegend=True
    )
    
    fig.update_yaxes(range=[0.23, 0.46])

    fig.show()
    
    
def create_stack_regressor(model_selection, final_estimator, features_by_importance):
    estimators = []
    params, _, _ = load_models_values()

    for key in model_selection:
        n = model_selection[key]
        feat_sel = ColumnTransformer(
            [(f"sel_{str(n)}", "passthrough", features_by_importance[:n])]
        )
        
        model = MODELS[key]['model']
        base_params = MODELS[key]['base_params']
        best_params = params[key][str(n)]
        estimators.append((key, Pipeline(
            [
                ('select', feat_sel),
                (key, model(**{**base_params, **best_params}))
            ]
        )))
    
    return StackingRegressor(
        estimators,
        final_estimator=final_estimator,
        cv=5
    )
    
def get_oof_predictions(model_selection, X, y, X_val):
    params, _, _ = load_models_values()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=72)
    oof_predictions = np.zeros((X.shape[0], len(model_selection)))
    oof_predictions_val = np.zeros((X_val.shape[0], len(model_selection)))    

    for i, key in enumerate(model_selection):
        for (train_index, holdout_index), (_, holdout_index_val) in zip(kf.split(X, y), kf.split(X_val)):
            n = model_selection[key]
            model = MODELS[key]['model']
            base_params = MODELS[key]['base_params']
            best_params = params[key][str(n)]
            
            instance = model(**{**base_params, **best_params})
            instance.fit(X.iloc[train_index], y.iloc[train_index])
            y_pred = instance.predict(X.iloc[holdout_index])
            y_pred_val = instance.predict(X_val.iloc[holdout_index_val])
            
            oof_predictions[holdout_index, i] = y_pred
            oof_predictions_val[holdout_index_val, i] = y_pred_val

    return pd.DataFrame(oof_predictions), pd.DataFrame(oof_predictions_val)

def make_stacking_objective_function(model_entity, search_space, model_selection, features_by_importance, X, y, cv, scoring="neg_mean_squared_error"):
    """
    Returns an Optuna objective function for a given final_model and parameter search space for a stacking regressor.
    """
    def objective(trial):
        params = model_entity['base_params']

        for pname, spec in search_space.items():
            if spec["type"] == "int":
                params[pname] = trial.suggest_int(
                    pname, spec["low_limit"], spec["top_limit"], log=spec.get("log", False)
                )
            elif spec["type"] == "float":
                params[pname] = trial.suggest_float(
                    pname, spec["low_limit"], spec["top_limit"], log=spec.get("log", False)
                )
            elif spec["type"] == "categorical":
                params[pname] = trial.suggest_categorical(pname, spec["choices"])
            else:
                raise ValueError(f"Unsupported param type {spec['type']}")

        print('a1')
        model = create_stack_regressor(model_selection, model_entity['model'](**params), features_by_importance)
        print('a2')
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        print('a3')
        return np.sqrt(-scores.mean())

    return objective

def plot_stacking_scores(cv_scores, val_scores, mode="mean", items_per_row=2):
    """
    mode = "mean"  -> plotta media/std delle fold + validation
    mode = "folds" -> plotta singole fold + validation
    """
    n_models = len(cv_scores)
    n_rows = (n_models + items_per_row - 1) // items_per_row

    fig = sp.make_subplots(
        rows=n_rows,
        cols=items_per_row,
        subplot_titles=list(cv_scores.keys())
    )

    for i, model_name in enumerate(cv_scores.keys()):
        df = cv_scores[model_name]

        row = i // items_per_row + 1
        col = i % items_per_row + 1

        if mode == "mean":
            # calcola media e std delle fold
            mean_score = df[0].mean()
            std_score = df[0].std()

            fig.add_trace(
                go.Bar(
                    x=["CV mean"],
                    y=[mean_score],
                    error_y=dict(type='data', array=[std_score]),
                    name=f"{model_name} CV",
                ),
                row=row, col=col
            )

            # aggiungi validation
            fig.add_trace(
                go.Bar(
                    x=["Validation"],
                    y=[val_scores[model_name]],
                    name=f"{model_name} Validation",
                ),
                row=row, col=col
            )

        elif mode == "folds":
            # punteggi singoli fold
            fig.add_trace(
                go.Bar(
                    x=[f"fold {j+1}" for j in range(len(df))],
                    y=df[0],
                    name=f"{model_name} CV folds",
                ),
                row=row, col=col
            )

            # aggiungi validation (allineato come ultima colonna)
            fig.add_trace(
                go.Bar(
                    x=["Validation"],
                    y=[val_scores[model_name]],
                    name=f"{model_name} Validation",
                ),
                row=row, col=col
            )

        else:
            raise ValueError("mode deve essere 'mean' o 'folds'")
        
    width = 500 if mode == 'folds' else 250

    fig.update_layout(
        barmode='group',
        height=300 * n_rows,
        width=width * items_per_row,
        showlegend=True
    )

    fig.update_yaxes(range=[0.23, 0.46])
    fig.show()
