from src.session.session import Session

from src.pipeline.demo_pipeline import demo_data_loading, demo_feature_engineering
from src.pipeline.pipeline import (
    ensure_dataset,
    initialise_models,
    run_model_comparison,
    run_classifier_workflow,
    run_feature_importance_workflow,
    run_model_optimisation_workflow
)

from src.evaluation.feature_subset_experiment import feature_subset_workflow
from src.features.feature_extraction import show_selected_features
from src.models.model_manager import describe_model

session = Session()
ensure_dataset(session)
initialise_models(session)

def select_active_model(session):
    if not session.models:
        print("No models available.")
        return

    print("\nAvailable models:")
    for i, name in enumerate(session.models.keys(), 1):
        print(f"{i}. {name}")

    choice = input("Select model: ")

    try:
        selected_name = list(session.models.keys())[int(choice) - 1]
        session.model = session.models[selected_name]
        print(f"Active model set to: {selected_name}")
    except:
        print("Invalid selection.")

def main_menu():
    print("\nMain Menu")
    print("1. Experiment Mode")
    print("2. URL Classifier")
    print("0. Exit")
    return input("Select option: ")


def experiment_menu():
    print("\n--- Experiment Mode ---")
    print("1. Load/inspect dataset")
    print("2. Build feature matrix")
    print("3. Train single model")
    print("4. Run full experiment")
    print("5. Feature Importance")
    print("6. Feature Subset Experiment")
    print("7. Model Optimisation")
    print("8. Evaluate Model")
    print("9. Model Info")
    print("10. Current Features")
    print("0. Back")
    return input("Select option: ")


def classifier_menu():
    print("\n--- URL Classifier ---")
    print("1. Select Model")  
    print("2. Classify URL")
    print("3. Model Info")
    print("0. Back")
    return input("Select option: ")

def model_menu():
    print("\n--- Model Experiments ---")
    print("1. Run Model Comparison")  
    print("2. Optimise Model")
    print("0. Back")
    return input("Select option: ")

def feature_menu():
    print("\n--- Feature Experiments ---")
    print("1. Feature importance")
    print("2. Feature subset experiment")
    print("3. Current features")
    print("0. Back")
    return input("Select option: ")

def data_menu():
    print("\n--- Data / Debug ---")
    print("1. Load/inspect dataset")
    print("2. Build feature matrix")
    print("0. Back")
    return input("Select option: ")

def run_experiment_mode():
    while True:
        print("\n--- Experiment Mode ---")
        print("1. Model Experiments")
        print("2. Feature Experiments")
        print("3. Data / Debug Tools")
        print("0. Back")

        choice = input("Select option: ")

        if choice == "1":
            run_model_experiments()

        elif choice == "2":
            run_feature_experiments()

        elif choice == "3":
            run_data_tools()

        elif choice == "0":
            break

def run_model_experiments():
    while True:
        choice = model_menu()

        if choice == "1":
            run_model_comparison(session) 

        elif choice == "2":
            run_model_optimisation_workflow(session)

        elif choice == "0":
            break

def run_feature_experiments():
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

def run_classifier_mode():
    while True:
        choice = classifier_menu()

        if choice == "1":
            select_active_model(session)

        elif choice == "2":
            run_classifier_workflow(session)

        elif choice == "3":
            describe_model(session.model)

        elif choice == "0":
            break

while True:
    choice = main_menu()

    if choice == "1":
        run_experiment_mode()

    elif choice == "2":
        run_classifier_mode()

    elif choice == "0":
        print("Exiting...")
        break

