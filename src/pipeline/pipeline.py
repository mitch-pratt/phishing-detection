import numpy as np

from src.evaluation.feature_subset_experiment import feature_subset_workflow
from src.models.model_manager import describe_model, predict_url
from src.models.grid_search import optimise_knn, optimise_nn, optimise_rf


from src.features.feature_extraction import extract_features, show_selected_features


from src.evaluation.feature_importance import run_feature_importance
from src.evaluation.metrics import evaluate_classification, plot_confusion_matrix
from src.evaluation.plots import visualise_results

from src.interface.cli import (
    classifier_menu,
    data_menu,
    feature_menu,
    model_menu,
    select_active_model
)


from src.pipeline.demo_pipeline import demo_data_loading, demo_feature_engineering


from src.pipeline.data_utils import (
    ensure_dataset,
    get_feature_data,
    apply_features
)
from src.session.session import Session








    
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

        if name == "K-Means":
            clusters = model.predict(X_test)
            y_pred = np.array([model.cluster_label_map[c] for c in clusters])
        else:
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

def run_experiment_mode(session):
    while True:
        print("\n--- Experiment Mode ---")
        print("1. Model Experiments")
        print("2. Feature Experiments")
        print("3. Data / Debug Tools")
        print("0. Back")

        choice = input("Select option: ")

        if choice == "1":
            run_model_experiments(session)

        elif choice == "2":
            run_feature_experiments(session)

        elif choice == "3":
            run_data_tools()

        elif choice == "0":
            break

def run_model_experiments(session):
    while True:
        choice = model_menu()

        if choice == "1":
            run_model_comparison(session) 

        elif choice == "2":
            run_model_optimisation_workflow(session)

        elif choice == "0":
            break

def run_feature_experiments(session):
    while True:
        choice = feature_menu()

        if choice == "1":
            run_feature_importance_workflow(session)

        elif choice == "2":
            feature_subset_workflow(session)

        elif choice == "3":
            show_selected_features(session)

        elif choice == "0":
            break

def run_data_tools():
    while True:
        choice = data_menu()

        if choice == "1":
            demo_data_loading()

        elif choice == "2":
            demo_feature_engineering()

        elif choice == "0":
            break

def run_classifier_mode(session):
    while True:
        choice = classifier_menu()

        if choice == "1":
            select_active_model(session)

        elif choice == "2":
            run_classifier_workflow(session)

        elif choice == "3":
            if session.model is None:
                print("No active model selected.")
            else:
                describe_model(session.model)

        elif choice == "0":
            break
