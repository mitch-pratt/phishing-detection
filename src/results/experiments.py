from src.results.metrics import evaluate_classification, plot_confusion_matrix
import numpy as np

def run_single_model(train_fn, X_train, X_test, y_train, y_test, evaluate_fn):
    model = train_fn(X_train, y_train)
    if hasattr(model, "cluster_centers_"):  # KMeans check
        y_pred = model.predict(X_test)
        y_pred = np.array([model.cluster_label_map[c] for c in y_pred])
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
            y_pred = np.array([model.cluster_label_map[c] for c in y_pred])

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

def random_forest_tree_experiment(X_train, X_test, y_train, y_test, tree_values):
    from sklearn.ensemble import RandomForestClassifier
    import time

    results = []

    for n in tree_values:
        print(f"\nTesting Random Forest with {n} trees")

        start_train = time.time()
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        train_time = time.time() - start_train

        start_pred = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_pred

        metrics = evaluate_classification(y_test, y_pred)

        results.append({
            "trees": n,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "train_time": train_time,
            "predict_time": pred_time
        })

    return results

from sklearn.ensemble import RandomForestClassifier
from src.results.metrics import evaluate_classification

def rf_feature_subset_experiment(X_train, X_test, y_train, y_test, ranked_indices):

    subset_sizes = [5, 10, 15, 20, len(ranked_indices)]
    results = []

    for size in subset_sizes:

        print(f"\nTesting top {size} features")

        selected = ranked_indices[:size]

        X_train_sub = X_train[:, selected]
        X_test_sub = X_test[:, selected]

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train_sub, y_train)

        y_pred = model.predict(X_test_sub)

        metrics = evaluate_classification(y_test, y_pred)

        results.append({
            "features_used": size,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        })

    return results



