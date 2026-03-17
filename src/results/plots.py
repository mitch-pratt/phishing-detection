import matplotlib.pyplot as plt
import numpy as np

def plot_metric(results, metric):
    if metric not in next(iter(results.values())):
        raise ValueError(f"Metric '{metric}' not found in results")
    models = list(results.keys())
    values = [results[m][metric] for m in models]

    plt.bar(models, values)
    plt.ylabel(metric.capitalize())
    plt.title(f"Model Comparison by {metric.capitalize()}")
    plt.ylim(0, 1)
    plt.show()

def plot_metric_select(results, metric, models_to_plot=None):
    #just plot all models if none selected
    if models_to_plot is None:
        models = list(results.keys())
    else:
        models = models_to_plot

    if metric not in next(iter(results.values())):
        raise ValueError(f"Metric '{metric}' not found in results")
    values = [results[m][metric] for m in models]

    plt.bar(models, values)
    plt.ylabel(metric.capitalize())
    plt.title(f"Model Comparison by {metric.capitalize()}")
    plt.ylim(0, 1)
    plt.show()

def plot_accuracy(results):
    print("Plotting accuracy...")
    model_names = []
    accuracies = []
    for model_name in results:
        model_names.append(model_name)
        accuracies.append(results[model_name]["accuracy"])
    plt.figure()
    plt.bar(model_names, accuracies)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.show(block=True)

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    print("Random Forest Feature Importance:")
    print("--------------------------------")

    for feature, importance in zip(sorted_features, sorted_importances):
        print(f"{feature}: {importance:.4f}")

    plt.figure()
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(len(sorted_importances)), sorted_importances)
    plt.xticks(range(len(sorted_importances)), sorted_features, rotation=90)
    plt.tight_layout()
    plt.show()

def plot_kmeans_feature_separation(model, feature_names, top_n=10):
    centers = model.cluster_centers_
    separation = np.var(centers, axis=0)

    indices = np.argsort(separation)[::-1]

    sorted_features = [feature_names[i] for i in indices][:top_n]
    sorted_separation = separation[indices][:top_n]

    plt.figure()
    plt.title("K-Means Feature Separation (Centroid Variance)")
    plt.bar(range(len(sorted_separation)), sorted_separation)
    plt.xticks(range(len(sorted_separation)), sorted_features, rotation=45, ha="right")
    plt.ylabel("Variance Across Cluster Centroids")
    plt.tight_layout()
    plt.show()