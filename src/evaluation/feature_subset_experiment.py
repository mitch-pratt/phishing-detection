import numpy as np
from sklearn.feature_selection import RFECV

from sklearn.base import is_classifier

from src.config.config import feature_names
from src.evaluation.metrics import evaluate_classification
from src.pipeline.pipeline import apply_features, build_and_split, ensure_dataset, initialise_models

def run_feature_subset_experiment(model, X_train, X_test, y_train, y_test):

    if model is None:
        raise ValueError("Train Random Forest")

    if not is_classifier(model):
        raise ValueError("Requires a supervised classifier")
    
    if not hasattr(model, "feature_importances_"):
        print("Feature subset experiment only supports Random Forest.")
        return None, None, None, None

    base_model = type(model)(**model.get_params())

    rfecv = RFECV(
        estimator=base_model,
        step=1,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    rfecv.fit(X_train, y_train)

    ranking = np.argsort(rfecv.ranking_)

    results = []
    best_result = None
    best_indices = None

    print("\nRunning feature subset experiment...\n")

    for k in range(1, X_train.shape[1] + 1):

        selected_idx = ranking[:k]

        X_train_sub = X_train[:, selected_idx]
        X_test_sub = X_test[:, selected_idx]

        model_sub = type(model)(**model.get_params())
        model_sub.fit(X_train_sub, y_train)

        y_pred = model_sub.predict(X_test_sub)
        metrics = evaluate_classification(y_test, y_pred)

        results.append({
            "num_features": k,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"]
        })

        print(f"{k} features → Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

        if best_result is None or metrics["accuracy"] > best_result["accuracy"]:
            best_result = results[-1]
            best_indices = selected_idx

    selected_feature_names = [feature_names[i] for i in best_indices]

    print("\nBest feature subset:")
    for f in selected_feature_names:
        print("-", f)

    print(f"\nBest accuracy: {best_result['accuracy']:.4f} using {best_result['num_features']} features")

    return results, ranking, best_indices, selected_feature_names

def find_min_features(results, threshold=0.90):
    for r in results:
        if r["accuracy"] >= threshold:
            return r
    return None

def feature_subset_workflow(session):

    ensure_dataset(session)

    try:
        model = session.models["Random Forest"]
    except KeyError:
        print("Error: Random Forest model not found in session.")
        return

    results, ranking, best_indices, selected_feature_names = run_feature_subset_experiment(
        model,
        session.X_train,
        session.X_test,
        session.y_train,
        session.y_test
    )

    if results is None:
        return

    best = find_min_features(results, threshold=0.90)

    if best:
        print(f"\nMinimum features for ≥90% accuracy: {best['num_features']}")
        print(f"Accuracy: {best['accuracy']:.4f}")

        use = input("\nApply this feature subset? (y/n): ")

        if use.lower() == "y":

            print("\n1. Use minimal subset")
            print("2. Use best accuracy subset")
            choice = input("Select option (1/2): ").strip()

            if choice == "1":
                selected_idx = ranking[:best["num_features"]]
                print(f"\nApplying minimal subset ({len(selected_idx)} features)...")
            else:
                selected_idx = best_indices
                print(f"\nApplying best subset ({len(selected_idx)} features)...")

            session.X_train = session.X_train[:, selected_idx]
            session.X_test = session.X_test[:, selected_idx]

            session.selected_features = selected_idx

            retrain_session_models(session)

            print("\nFeature subset applied.")
            print(f"{len(selected_idx)} features now active.")
            print("\nSelected features:")
            for i in selected_idx:
                print("-", feature_names[i])

        else:
            print("Feature subset not applied.")

    else:
        print("\nNo subset reached 90% accuracy.")

def retrain_session_models(session):
    
    X_train = apply_features(session, session.X_train)
    y_train = session.y_train

    for name, model in session.models.items():
        print(f"Retraining {name} on current feature subset...")
        model.fit(X_train, y_train)