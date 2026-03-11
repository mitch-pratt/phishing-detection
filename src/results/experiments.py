from src.results.metrics import evaluate_classification, plot_confusion_matrix
import numpy as np

def run_single_model(train_fn, X_train, X_test, y_train, y_test, evaluate_fn):
    model = train_fn(X_train, y_train)
    if hasattr(model, "cluster_centers_"):  # KMeans check
        y_pred = model.predict(X_test)
        y_pred = map_clusters_to_labels(y_test, y_pred)
    else:
        y_pred = model.predict(X_test)
    metrics = evaluate_fn(y_test, y_pred)
    return model, metrics

def run_experiment(models, X_train, X_test, y_train, y_test, evaluate_fn):
    results = {}
    trained_models = {}

    for name, train_fn in models.items():
        model = train_fn(X_train, y_train)

        if hasattr(model, "cluster_centers_"):
            y_pred = model.predict(X_test)
            y_pred = map_clusters_to_labels(y_test, y_pred)

        else:
            y_pred = model.predict(X_test)

        results[name] = evaluate_fn(y_test, y_pred)
        trained_models[name] = model
    
    return trained_models, results

def map_clusters_to_labels(y_true, clusters):
    mapped = np.zeros_like(clusters)

    for cluster in np.unique(clusters):
        mask = clusters == cluster

        majority_label = np.bincount(y_true[mask]).argmax()
        mapped[mask] = majority_label

    return mapped



