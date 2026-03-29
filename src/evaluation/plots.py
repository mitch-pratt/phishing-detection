import matplotlib.pyplot as plt
from src.config.config import feature_names
import numpy as np

def plot_model_comparison(results):
    import numpy as np
    import matplotlib.pyplot as plt

    models = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    x = np.arange(len(models))
    width = 0.2

    plt.figure()

    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in models]
        plt.bar(x + i * width, values, width, label=metric)

    plt.xticks(x + width * (len(metrics) - 1) / 2, models)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()
    plt.title(title)
    plt.tight_layout()
    plt.show()

def show_feature_summary(session, feature_names):
    print("\n=== Feature Summary ===")

    if session.selected_features is None:
        print("Using FULL feature set")
        for f in feature_names:
            print("-", f)
    else:
        print(f"Using REDUCED feature set ({len(session.selected_features)} features):")
        for i in session.selected_features:
            print("-", feature_names[i])

from src.config.config import feature_names

def visualise_results(session, results, X_test=None):

    if input("Visualise results? (y/n): ") != "y":
        return

    plot_model_comparison(results)

    if X_test is not None:
        show_cm = input("Show confusion matrices? (y/n): ")
        if show_cm == "y":

            for name, model in session.models.items():

                if not hasattr(model, "predict"):
                    continue

                try:
                    y_pred = model.predict(X_test)

                    print(f"\nConfusion Matrix: {name}")
                    plot_confusion_matrix(session.y_test, y_pred)

                except Exception as e:
                    print(f"Skipping {name}: {e}")


    show_feats = input("Show feature summary? (y/n): ")
    if show_feats == "y":
        show_feature_summary(session, feature_names)

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    print("Random Forest Feature Importance:")
    print("--------------------------------")

    for feature, importance in zip(sorted_features, sorted_importances):
        print(f"{feature}: {importance:.2f}")

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