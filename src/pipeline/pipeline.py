from src import models
from src.config.config import models, MODEL_MENU
from src.evaluation.feature_importance import run_feature_importance
from src.evaluation.metrics import evaluate_classification, plot_confusion_matrix
from src.evaluation.plots import visualise_results
from src.evaluation.runner import run_experiment
from src.features.feature_extraction import build_feature_matrix, extract_features, show_selected_features

from src.dataset.dataset_manager import DataLoader
from src.interface.cli import show_feature_menu
from src.models.grid_search import optimise_knn, optimise_nn, optimise_rf
from src.models.model_manager import predict_url

from sklearn.model_selection import train_test_split
from src.config.config import feature_names

def initialise_models(session):
 
    if session.X_train is None:
        session.X_train, session.X_test, session.y_train, session.y_test = build_and_split()

    X_train = apply_features(session, session.X_train)
    X_test = apply_features(session, session.X_test)
    y_train = session.y_train
    y_test = session.y_test

    base_models = {
        "Random Forest": models["Random Forest"],
        "KNN": models["KNN"],
        "Neural Network": models["Neural Network"],
        "K-Means": models["K-Means"]
    }

    print("Training base models...")

    for name, train_fn in base_models.items():
        print(f"\nTraining {name}...")
    
        model = train_fn(X_train, y_train)
        
        
        session.models[name] = model

        y_pred = model.predict(X_test)
        metrics = evaluate_classification(y_test, y_pred)
        print(f"{name} trained. Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    print("\nAll base models trained and stored in session.")

def feature_split(X, y, test_size=0.25, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def ensure_dataset(session):
    if session.X_train is None:
        print("Building features automatically...")
        session.X_train, session.X_test, session.y_train, session.y_test = build_and_split()

        session.selected_features = list(range(session.X_train.shape[1]))

def get_feature_data(session):
    X_train = apply_features(session, session.X_train)
    X_test = apply_features(session, session.X_test)
    return X_train, X_test

def build_and_split():
    
    loader = DataLoader("data/urlset.csv")
    df = loader.load_csv()
    df = loader.clean()
    urls, labels = loader.get_urls_labels()
    
    X, y = build_feature_matrix(urls, labels)
    X_train, X_test, y_train, y_test = feature_split(X, y)

    return X_train, X_test, y_train, y_test

def apply_features(session, features):
    import numpy as np

    features = np.array(features)

    if session.selected_features is None:
        return features

    selected_idx = session.selected_features

    if features.ndim == 1:
        return features[selected_idx]

    elif features.ndim == 2:
        return features[:, selected_idx]

    else:
        raise ValueError(f"Unexpected features shape: {features.shape}")
    
def run_model_comparison(session):

    ensure_dataset(session)

    if not session.models:
        print("No models available. Initialise models first.")
        return

    print("\nAvailable models:")
    model_names = list(session.models.keys())

    for i, name in enumerate(model_names, 1):
        print(f"{i}. {name}")

    print("(comma separated for multiple, e.g. 1,2,3)")
    choices = input("Choice: ").split(",")

    selected_models = {}
    for c in choices:
        c = c.strip()
        if c.isdigit() and 1 <= int(c) <= len(model_names):
            name = model_names[int(c) - 1]
            selected_models[name] = session.models[name]

    if not selected_models:
        print("No valid models selected.")
        return

    X_train, X_test = get_feature_data(session)
    y_test = session.y_test

    print("\nActive features for this comparison:")
    show_selected_features(session)
    print("\nRunning model comparison...\n")

    results = {}

    for name, model in selected_models.items():
        print(f"Evaluating {name}...")

        y_pred = model.predict(X_test)
        metrics = evaluate_classification(y_test, y_pred)

        results[name] = metrics


    print("\n=== Results ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    visualise_results(session, results, X_test)

def run_classifier_workflow(session):

    if session.model is None:
        print("Train a model first.")
        return

    url = input("Enter URL: ")

    features = extract_features(url)
    features = apply_features(session, features)

    prediction = predict_url(session.model, features)

    print("Prediction:", {0: "Benign", 1: "Malicious"}.get(prediction))

def run_feature_importance_workflow(session):
    print("\nSelect feature importance method:")
    print("1. Random Forest")
    print("2. K-Means Clustering")
    
    choice = input("Select method: ")
    run_feature_importance(choice, session)

def run_model_optimisation_workflow(session):

    ensure_dataset(session)
    X_train, _ = get_feature_data(session)
    y_train = session.y_train

    print("\nSelect model to optimise:")
    print("1. KNN")
    print("2. Random Forest")
    print("3. Neural Network")

    choice = input("Choice: ")

    if choice == "1":
        name = "KNN"
        model, params, score = optimise_knn(X_train, y_train)

    elif choice == "2":
        name = "Random Forest"
        model, params, score = optimise_rf(X_train, y_train)

    elif choice == "3":
        name = "Neural Network"
        model, params, score = optimise_nn(X_train, y_train)

    else:
        print("Invalid selection.")
        return

    print(f"\nBest Parameters for {name}: {params}")
    print(f"Best CV Score: {score:.4f}")

    if input(f"Replace existing {name} model with optimised version? (y/n): ") == "y":
        session.models[name] = model
        print(f"{name} model updated in session.")
    else:
        print("Optimised model discarded.")

def run_model_evaluation_workflow(session):

    if session.model is None:
            print("No model available. Train or optimise first.")
            return

    print(f"Evaluating model: {type(session.model).__name__}")

    X_test = apply_features(session, session.X_test)

    y_pred = session.model.predict(X_test)

    metrics = evaluate_classification(session.y_test, y_pred)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    show_plots = input("Show confusion matrix? (y/n): ")
    if show_plots == "y":
        plot_confusion_matrix(session.y_test, y_pred, "Confusion Matrix")