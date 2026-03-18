from src.results.metrics import evaluate_classification, plot_confusion_matrix
import numpy as np

def run_single_model_with_display(name, train_fn, X_train, X_test, y_train, y_test):
    print(f"\nTraining {name}...")
    model, metrics = run_single_model(
        train_fn,
        X_train, X_test,
        y_train, y_test,
        evaluate_classification
    )
    
    
    return model #, metrics

def run_single_model(train_fn, X_train, X_test, y_train, y_test, evaluate_fn):
    model = train_fn(X_train, y_train)
    if hasattr(model, "cluster_centers_"): 
        y_pred = model.predict(X_test)
        y_pred = np.array([model.cluster_label_map[c] for c in y_pred])
    else:
        y_pred = model.predict(X_test)
    metrics = evaluate_fn(y_test, y_pred)
    return model, metrics



def map_clusters_to_labels(y_true, clusters):
    mapped = np.zeros_like(clusters)

    for cluster in np.unique(clusters):
        mask = clusters == cluster

        majority_label = np.bincount(y_true[mask]).argmax()
        mapped[mask] = majority_label

    return mapped







