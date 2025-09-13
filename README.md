Development of Machine Learning model for [Ames Housing Kaggle competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

Final submission: 0.11744, top 2%.

## Project Structure
```
├─ 🚩 checkpoints  -> scores and parameters of previous meaningful runs
├─ 📊 data         -> given and preprocessed data
├─ 📓 notebooks    
|  ├─ pre-processing.ipynb   -> data exploration, feature engineering, feature transformation, normalization
|  └─ predictions.ipynb      -> feature selection, model optimization
└─ 📚 src
   ├─ data.py           -> loading data, preparing submission
   ├─ preprocessing.py  -> missing values, data exploration, feature engineering, data trasformation, normalization, outliers
   ├─ features.py       -> feature importance and feature selection
   ├─ models.py         -> model optimization and evaluation
   └─ predict.py        -> make predictions
```
