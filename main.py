from src.models.neural_network import train_neural_network
from src.models.knn_model import train_knn
from src.models.k_means import train_kmeans
from src.features.feature_split import feature_split
from src.results.metrics import plot_confusion_matrix, evaluate_classification
from src.utils.data_loader import DataLoader
from src.utils.data_checker import DataChecker
from src.features.feature_matrix import build_feature_matrix, FeatureMatrixChecker
from src.models.random_forest import train_random_forest, predict
from src.results.plots import plot_metric, plot_metric_select
from src.results.experiments import run_experiment, run_single_model, map_clusters_to_labels
from src.features.feature_extraction import extract_features
import numpy as np
from src.results.plots import plot_kmeans_feature_separation, plot_feature_importance

feature_names = [
    "URL length",
    "Dot count",
    "Hyphen count",
    "Digit count",
    "@ symbol present",
    "Subdomain count",

    "Suspicious word count",

    "Raw word count",
    "Average word length",
    "Longest word length",
    "Shortest word length",
    "Word length std",
    "Digit ratio",

    "Domain length",
    "Path length",
    "Path level",

    "Dash count in hostname",

    "Special char count",
    "Underscore count",
    "Percent encoding count",
    "Ampersand count",
    "Hash count",

    "Query component count",

    "No HTTPS",
    "IP address in URL",
    "HTTPS token in hostname",

    "Domain keyword in subdomain",

    "Known TLD",

    "Consecutive character repeat",
    "Punycode",

    "Contains WWW",
    "Contains .com"
]

X_train = X_test = y_train = y_test = None
current_model = None

models = {
    "Random Forest": train_random_forest,
    "KNN": train_knn,
    "Neural Network": train_neural_network,
    "K-Means": train_kmeans
}

MODEL_MENU = {
    "1": "Random Forest",
    "2": "KNN",
    "3": "Neural Network",
    "4": "K-Means"
}


def show_menu():
    print("1. Load/inspect dataset")
    print("2. Build feature matrix")
    print("3. Train single model")
    print("4. Run full experiment")
    print("5. URL Classifier")
    print("6. Feature Importance")
    print("0. Exit")

def show_model_menu():
    print("Select model:")
    print("1. Random Forest")
    print("2. KNN")
    print("3. Neural Network")
    print("4. K-Means")

def show_feature_menu():
    print("Select feature importance method:")
    print("1. Random Forest")
    print("2. K-Means Clustering")

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

def demo_data_loading():
    print("\n Loading dataset...")
    loader = DataLoader("data/urlset.csv")
    df = loader.load_csv()
    print(f"Raw dataset shape: {df.shape}")
    df = loader.clean()
    print(f"Cleaned dataset shape: {df.shape}")
    urls, labels = loader.get_urls_labels()
    print("\n Dataset summary")
    DataChecker.print_summary(urls, labels)

def demo_feature_engineering():
    
    loader = DataLoader("data/urlset.csv")
    df = loader.load_csv()
    df = loader.clean()
    urls, labels = loader.get_urls_labels()
    sample_url = urls[0]
    features = extract_features(sample_url)
    print("\n Feature extraction example")
    print("Sample URL:", sample_url)
    print("Extracted features:", features)
    

    for name, value in zip(feature_names, features):
        print(f"{name}: {value}")

    print("\n Building feature matrix")
    X, y = build_feature_matrix(urls, labels)
    FeatureMatrixChecker.print_summary(X, y)

    print("\n Train/test split")
    X_train, X_test, y_train, y_test = feature_split(X, y)

    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    return X_train, X_test, y_train, y_test

def run_single_model_with_display(name, train_fn, X_train, X_test, y_train, y_test):
    print(f"\nTraining {name}...")
    model, metrics = run_single_model(
        train_fn,
        X_train, X_test,
        y_train, y_test,
        evaluate_classification
    )
    plot_confusion_matrix(y_test, model.predict(X_test), name)
    return model #, metrics

while True:

    show_menu()
    choice = input("Select option: ")

    if choice == "1":
        demo_data_loading()
    elif choice == "2":
        X_train, X_test, y_train, y_test = demo_feature_engineering()
    elif choice == "3":
        if X_train is None:
            print("Please build feature matrix first (option 2).")
        else:
            show_model_menu()
            model_choice = input("Select model: ")
            selected_name = MODEL_MENU.get(model_choice)
            if selected_name is None:
                print("Invalid selection")
            else:
                train_fn = models[selected_name]
                current_model = run_single_model_with_display(
                    selected_name,
                    train_fn,
                    X_train, X_test,
                    y_train, y_test
                )
    elif choice == "4":
        if X_train is None:
            print("Please build feature matrix first (option 2).")
        else:
            print("\nRunning experiment...")
            trained_models, results = run_experiment(
            models,
            X_train, X_test,
            y_train, y_test,
            evaluate_classification
        )
            
            for metric in ["accuracy", "precision", "recall", "f1"]:
                plot_metric_select(results, metric)

    elif choice == "5":
        print(set(y_train))
        print(extract_features("www.google.com"))
        print(extract_features("nobell.it/..."))
        print("Training samples:", len(X_train))
        y_pred = current_model.predict(X_test)
        print(np.unique(y_pred, return_counts=True))
        if current_model is None:
            print("Please train a model first (option 3).")
        else:
            url = input("Enter URL: ")
            features = extract_features(url)

            if hasattr(current_model, "cluster_centers_"):
                cluster = current_model.predict([features])[0]
                prediction = current_model.cluster_label_map[cluster]
            else:
                prediction = current_model.predict([features])
        
            print("Prediction:", prediction)

    elif choice == "6":
        if X_train is None:
            print("Please build feature matrix first (option 2).")

        else:
            show_feature_menu()
            feature_imp_choice = input("Select method: ")

            run_feature_importance(
                feature_imp_choice,
                X_train,
                y_train,
                X_test,
                feature_names
            )

    elif choice == "0":
        print("Exiting...")
        break
    


"""

loader.clean()
urls, labels = loader.get_urls_labels()

#DataChecker.print_summary(urls, labels)










plot_metric_select(results, "f1", ["Random Forest", "Neural Network"])






plot_metric(results, "f1")
print_classification(y_test, y_pred)





"""