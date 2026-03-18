from src.results.metrics import evaluate_classification

def random_forest_tree_experiment(X_train, X_test, y_train, y_test, tree_values):
    from sklearn.ensemble import RandomForestClassifier
    import time

    results = []

    for n in tree_values:
        print(f"\nTesting Random Forest with {n} trees")

        start_train = time.time()
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        train_time = time.time() - start_train

        start_pred = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_pred

        metrics = evaluate_classification(y_test, y_pred)

        results.append({
            "trees": n,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "train_time": train_time,
            "predict_time": pred_time
        })

    return results