import numpy as np

def run_single_model(train_fn, X_train, X_test, y_train, y_test, evaluate_fn):
    model = train_fn(X_train, y_train)
    if hasattr(model, "cluster_centers_"): 
        y_pred = model.predict(X_test)
        y_pred = np.array([model.cluster_label_map[c] for c in y_pred])
    else:
        y_pred = model.predict(X_test)
    metrics = evaluate_fn(y_test, y_pred)
    return model, metrics








