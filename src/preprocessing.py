import math
import json
import numpy as np

import plotly.subplots as ps
import plotly.graph_objects as go


# 1. MISSING VALUES
# =================================================

def get_fill_na_values(df):
    lot_frontage_medians = vals = df.groupby('Neighborhood')['LotFrontage'].median()
    
    return {
        # Garage
        'GarageCond': 'na',
        'GarageQual': 'na',
        'GarageType': 'na',
        'GarageYrBlt': df['YearBuilt'],
        'GarageFinish': 'na',
        'GarageArea': 0,
        'GarageCars': 0,
        
        # Basement
        'BsmtFinType1': 'na',
        'BsmtFinType2': 'na',
        'BsmtExposure': 'na',
        'BsmtCond': 'na',
        'BsmtQual': 'na',
        'TotalBsmtSF': 0,
        'BsmtHalfBath': 0,
        'BsmtFullBath': 0,
        'BsmtFinSF1': 0,
        'BsmtFinSF2': 0,
        'BsmtUnfSF': 0,
        
        # Masonry Veener
        'MasVnrType': 'na',
        'MasVnrArea': 0,
        
        # Fireplace
        'FireplaceQu': 'na',
        
        # Electrical
        'Electrical': 'SBrkr',
        
        # Pool 
        'PoolQC': 'na',
        'PoolArea': 0,
        
        # Miscellaneous features
        'MiscFeature': 'na',
        
        # Alley
        'Alley': 'na',
        
        # Fence
        'Fence': 'na',
        
        # LotFrontage
        'LotFrontage': df['Neighborhood'].map(lot_frontage_medians),
        
        #
        'MSZoning': 'RL',
        
        #
        'Functional': 'Typ',
        
        # Exterior
        'Exterior1st': 'VinylSd',
        'Exterior2nd': 'VinylSd',
        
        #
        'KitchenQual': 'TA',
        
        # 
        'SaleType': 'WD',
        
        #
        'Utilities': 'AllPub',
    }

def fill_na(df):
    df_aux = df.copy()
    fill_na_values = get_fill_na_values(df_aux)
    
    for key, val in fill_na_values.items():
        df_aux[key] = df_aux[key].fillna(val)
        
    return df_aux


# 2. DATA EXPLORATION
# =================================================

def hist_matrix(df):
    n_attr = len(df.columns)
    cols = 6
    rows = math.ceil(n_attr / 6)
    
    fig = ps.make_subplots(rows=rows, cols=cols, subplot_titles=df.columns)
    
    for i, attr in enumerate(df.columns):
        fig.add_trace(go.Histogram(x=df[attr]), row=i//cols+1, col=i%cols+1)
        
    fig.update_layout(
        height=rows*200,
        width=cols*200,
        showlegend=False,
        title_text='SalePrice investigation'
    )
        
    fig.show()

def plot_log(df, attributes, cols=1):
    n_attr = len(attributes)
    titles = np.array([[f'{a} Distribution', f'{a} Log Distribution'] for a in attributes]).flatten()
    
    fig = ps.make_subplots(rows=math.ceil(n_attr/cols), cols=2*cols, subplot_titles=titles)

    for i, attr in enumerate(attributes):
        row = i // cols + 1
        col = i % 2 * cols + 1
        
        fig.add_trace(
            go.Histogram(x=df[attr]),
            row=row, col=col,
        )

        fig.add_trace(
            go.Histogram(x=np.log1p(df[attr])),
            row=row, col=col+1    
        )

    fig.update_layout(
        height=100 + 150*n_attr/cols,
        width=100 + 500 * cols,
        showlegend=False,
        title_text=f'Log investigation'
    )
    
    fig.show()
    

# 3. FEATURE ENGINEERING
# =================================================

def engineer_features(df):
    dfe = df.copy()
    
    dfe['TotalSF'] = dfe['1stFlrSF'] + dfe['2ndFlrSF'] + dfe['TotalBsmtSF']
    
    dfe['TotalBath'] = (dfe['BsmtFullBath'] + 0.5 * dfe['BsmtHalfBath'] +
                        dfe['FullBath'] + 0.5 * dfe['HalfBath'])
    
    dfe['TotalPorchSF'] = (dfe['OpenPorchSF'] + dfe['3SsnPorch'] + dfe['EnclosedPorch'] + 
                           dfe['ScreenPorch'] + dfe['WoodDeckSF'])
    
    dfe['OverallGrade'] = dfe['OverallQual'] * dfe['OverallCond']
    dfe['QualArea'] = dfe['GrLivArea'] * dfe['OverallQual']
    dfe['CondArea'] = dfe['GrLivArea'] * dfe['OverallCond']
    
    dfe['HouseAge'] = dfe['YrSold'] - dfe['YearBuilt']
    dfe['RemodAge'] = dfe['YrSold'] - dfe['YearRemodAdd']
    
    dfe['HasBasement'] = (dfe['TotalBsmtSF'] > 0).astype(int)
    dfe['HasGarage'] = (dfe['GarageArea'] > 0).astype(int)
    dfe['HasFireplace'] = (dfe['Fireplaces'] > 0).astype(int)
    dfe['HasPool'] = (dfe['PoolArea'] > 0).astype(int)
    dfe['Has2ndFloor'] = (dfe['2ndFlrSF'] > 0).astype(int)
    
    return dfe
    

# 4. DATA TRANSFORMATION
# =================================================

def log_features(df, attributes):
    dfl = df.copy()
    
    for attr in attributes:
        dfl[attr] = np.log1p(dfl[attr])
        
    return dfl

def transform_quality_attributes(df):
    dft = df.copy()
    
    for attr in dft.columns:
        if 'Qual' in attr or 'QC' in attr or 'Cond' in attr or 'Qu' in attr:
            dft[attr] = dft[attr].replace('na', 0)
            dft[attr] = dft[attr].replace('Po', 1)
            dft[attr] = dft[attr].replace('Fa', 2)
            dft[attr] = dft[attr].replace('TA', 3)
            dft[attr] = dft[attr].replace('Gd', 4)
            dft[attr] = dft[attr].replace('Ex', 5)
            
    return dft
            
def get_transformation_values():
    return {
        'CentralAir': {
            'N': 0,
            'Y': 1,
        },
        'LotShape': {
            'Reg': 0,
            'IR1': 1,
            'IR2': 2,
            'IR3': 3,
        },
        'BsmtExposure': {
            'na': 0,
            'No': 1,
            'Mn': 2,
            'Av': 3,
            'Gd': 4,
        },
        'BsmtFinType1': {
            'na': 0,
            'Unf': 1,
            'LwQ': 2,
            'Rec': 3,
            'BLQ': 4,
            'ALQ': 5,
            'GLQ': 6,
        },
        'BsmtFinType2': {
            'na': 0,
            'Unf': 1,
            'LwQ': 2,
            'Rec': 3,
            'BLQ': 4,
            'ALQ': 5,
            'GLQ': 6,
        },
        'Functional': {
            'Sal': 0,
            'Sev': 1,
            'Maj2': 2,
            'Maj1': 3,
            'Mod': 4,
            'Min2': 5,
            'Min1': 6,
            'Typ': 7,
        },
        'Electrical': {
            'FuseP': 0,
            'FuseF': 1,
            'FuseA': 2,
            'Mix': 3,
            'SBrkr': 4,
        },
        'GarageFinish': {
            'na': 0,
            'Unf': 1,
            'RFn': 2,
            'Fin': 3,
        },
        'PavedDrive': {
            'N': 0,
            'P': 1,
            'Y': 2,
        },
    }
    
def transform_hardcoded_attributes(df):
    dft = df.copy()
    transformation_values = get_transformation_values()
    
    for attr, transf in transformation_values.items():
        for key, val in transf.items():
            dft[attr] = dft[attr].replace(key, val)
            
    return dft

def transform_attributes(df):
    dft = df.copy()
    
    dft = transform_quality_attributes(dft)
    dft = transform_hardcoded_attributes(dft)
    
    return dft
    

# 5. NORMALIZATION
# =================================================

def compute_norm_values_and_save(df, prefix):
    norm_values = { 'mean' : {}, 'std' : {}}
    
    for col in df.columns:
        norm_values['mean'][col] = df[col].mean()
        norm_values['std'][col] = df[col].std()
        
    with open(f'../data/{prefix}_normalization_values.json', 'w', encoding='utf-8') as f:
        json.dump(norm_values, f, ensure_ascii=False, indent=2)
        
def normalize(df, prefix):
    dfn = df.copy()
    with open(f'../data/{prefix}_normalization_values.json', 'r') as f:
        norm_values = json.load(f)
        
    for col in dfn.columns:
        dfn[col] = (dfn[col] - norm_values['mean'][col]) / norm_values['std'][col]
        
    return dfn
        
    