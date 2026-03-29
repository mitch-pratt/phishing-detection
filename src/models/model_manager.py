import numpy as np

def train_model(train_fn, X_train, y_train):
    return train_fn(X_train, y_train)

def predict_model(model, X):
    if hasattr(model, "cluster_centers_"):
        clusters = model.predict(X)
        return np.array([model.cluster_label_map[c] for c in clusters])
    
    return model.predict(X)

def evaluate_model(model, X_test, y_test, evaluate_fn):
    y_pred = model.predict(X_test)
    return evaluate_fn(y_test, y_pred)

def predict_url(model, features):
    import numpy as np

    features = np.array(features)

    if features.ndim == 1:
        features = features.reshape(1, -1)

    if hasattr(model, "cluster_centers_"):
        cluster = model.predict(features)[0]
        return model.cluster_label_map[cluster]

    return model.predict(features)[0]

def describe_model(model):
    if model is None:
        print("No model loaded.")
        return

    print("\n--- Current Model ---")
    print("Type:", type(model).__name__)

    print("\nParameters:")
    params = model.get_params()

    for key, value in params.items():
        print(f"{key}: {value}")