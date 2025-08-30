import pandas as pd

def get_raw_predictions(model_entity, params, X, y, X_test):
    model = model_entity['model'](**params)
    model.fit(X, y)
    
    return model.predict(X_test)
    