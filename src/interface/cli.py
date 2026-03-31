def show_menu():
    print("1. Load/inspect dataset")
    print("2. Build feature matrix")
    print("3. Train single model")
    print("4. Run full experiment")
    print("5. URL Classifier")
    print("6. Feature Importance")
    print("7. Model Optimisation")
    print("8. Evaluate Model")
    print("9. Model Info")
    print("10. Current Features")
    print("11. Feature Subset Experiment")
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