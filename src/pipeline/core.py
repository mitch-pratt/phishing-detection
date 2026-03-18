import numpy as np

def train_model(train_fn, X_train, y_train):
    return train_fn(X_train, y_train)

def predict_model(model, X):
    if hasattr(model, "cluster_centers_"):
        clusters = model.predict(X)
        return np.array([model.cluster_label_map[c] for c in clusters])
    
    return model.predict(X)

def evaluate_model(model, X_test, y_test, evaluate_fn):
    y_pred = predict_model(model, X_test)
    return evaluate_fn(y_test, y_pred)