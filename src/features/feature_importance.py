from src.models.k_means import train_kmeans
from src.models.random_forest import train_random_forest
from src.results.plots import plot_kmeans_feature_separation, plot_feature_importance
from src.utils.config import feature_names

def run_feature_importance(option, X_train, y_train, X_test, features_names):
    if option == "1":
        print("\nRunning Random Forest feature importance...")

        model = train_random_forest(X_train, y_train)
        plot_feature_importance(model, feature_names)

    elif option == "2":
        print("\nRunning K-Means clustering analysis...")

        model = train_kmeans(X_train, y_train)
        plot_kmeans_feature_separation(model, feature_names)

    else:
        print("Invalid selection.")