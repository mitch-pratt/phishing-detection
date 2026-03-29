from src.evaluation.kmeans_analysis import describe_clusters
from src.evaluation.plots import plot_feature_importance, plot_kmeans_feature_separation
from src.config.config import feature_names as all_feature_names  # import full feature names list

def run_feature_importance(option, session):
   
    X_train, _ = session.X_train, session.X_test
    y_train = session.y_train

    if session.selected_features is None:
        print("No features selected yet. Using full feature set.")
        session.selected_features = list(range(len(all_feature_names)))

    readable_feature_names = [all_feature_names[i] for i in session.selected_features]

    if option == "1":
        print("\nRunning Random Forest feature importance...")
        if "Random Forest" not in session.models:
            print("Random Forest model not found in session. Please train it first.")
            return
        model = session.models["Random Forest"]
        plot_feature_importance(model, readable_feature_names)

    elif option == "2":
        print("\nRunning K-Means clustering analysis...")
        if "K-Means" not in session.models:
            print("K-Means model not found in session. Please train it first.")
            return
        model = session.models["K-Means"]

        describe = input("Show cluster breakdown? (y/n): ")
        if describe.lower() == "y":
            describe_clusters(model, X_train, y_train, readable_feature_names)

        plot = input("Show feature separation plot? (y/n): ")
        if plot.lower() == "y":
            plot_kmeans_feature_separation(model, readable_feature_names)

    else:
        print("Invalid selection.")