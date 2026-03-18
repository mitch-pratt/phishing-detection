import numpy as np

from src.results.metrics import evaluate_classification
from src.results.plots import plot_metric_select
from src.results.experiments import run_single_model_with_display, plot_confusion_matrix
from src.experiments.runner import run_experiment

from src.features.feature_extraction import extract_features
from src.features.feature_importance import run_feature_importance

from src.utils.session import Session
from src.utils.demo_pipeline import demo_data_loading, demo_feature_engineering
from src.utils.config import models, MODEL_MENU, feature_names
from src.utils.cluster_check import predict_url

from src.ui.cli import show_menu, show_feature_menu, show_model_menu

session = Session()

while True:

    show_menu()
    choice = input("Select option: ")

    if choice == "1":
        demo_data_loading()

    elif choice == "2":
        session.X_train, session.X_test, session.y_train, session.y_test = demo_feature_engineering()

    elif choice == "3":
        if session.X_train is None:
            print("Building features automatically...")
            session.X_train, session.X_test, session.y_train, session.y_test = demo_feature_engineering()
        
        show_model_menu()
        model_choice = input("Select model: ")
        selected = MODEL_MENU.get(model_choice)
        if selected:
            session.model = run_single_model_with_display(
                selected,
                models[selected],
                session.X_train, 
                session.X_test,
                session.y_train, 
                session.y_test
            )
        show_plots = input("Show confusion matrix? (y/n): ")
        if show_plots == "y":
            plot_confusion_matrix(session.y_test, session.model.predict(session.X_test), "confusion matrix")

    elif choice == "4":
        if session.X_train is None:
            print("Building features automatically...")
            session.X_train, session.X_test, session.y_train, session.y_test = demo_feature_engineering()
        else:

            print("Select models to run (comma-separated):")
            print("1 RF, 2 KNN, 3 NN, 4 KMeans")

            choices = input("Choice: ").split(",")

            selected_models = {
                MODEL_MENU[c.strip()]: models[MODEL_MENU[c.strip()]]
                for c in choices if c.strip() in MODEL_MENU
            }

            print("\nRunning experiment...")
            trained_models, results = run_experiment(
            selected_models,
            session.X_train, 
            session.X_test,
            session.y_train, 
            session.y_test,
            evaluate_classification
        )
            
            print(results)

            plot = input("Show plots? (y/n): ")

            if plot == "y":
                for metric in ["accuracy", "precision", "recall", "f1"]:
                    plot_metric_select(results, metric)

    elif choice == "5":
        if session.model is None:
            print("Please train a model first (option 3).")
        else:
            print(f"Using model: {type(session.model).__name__}")
            url = input("Enter URL: ")
            features = extract_features(url)
            prediction = predict_url(session.model, features)
            label_map = {0: "Benign", 1: "Malicious"}
            print("Prediction:", label_map.get(prediction, prediction))

    elif choice == "6":
        if session.X_train is None:
            print("Building features automatically...")
            session.X_train, session.X_test, session.y_train, session.y_test = demo_feature_engineering()

        else:
            show_feature_menu()
            feature_imp_choice = input("Select method: ")

            run_feature_importance(
                feature_imp_choice,
                session.X_train,
                session.y_train,
                session.X_test,
                feature_names
            )

    elif choice == "0":
        print("Exiting...")
        break