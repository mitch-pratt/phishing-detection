from sklearn.ensemble import RandomForestClassifier
from src.results.metrics import evaluate_classification

def rf_feature_subset_experiment(X_train, X_test, y_train, y_test, ranked_indices):

    subset_sizes = [5, 10, 15, 20, len(ranked_indices)]
    results = []

    for size in subset_sizes:

        print(f"\nTesting top {size} features")

        selected = ranked_indices[:size]

        X_train_sub = X_train[:, selected]
        X_test_sub = X_test[:, selected]

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train_sub, y_train)

        y_pred = model.predict(X_test_sub)

        metrics = evaluate_classification(y_test, y_pred)

        results.append({
            "features_used": size,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        })

    return results