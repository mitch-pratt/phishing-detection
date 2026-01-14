from src.results.metrics import evaluate_classification, plot_confusion_matrix

def run_single_model(train_fn, X_train, X_test, y_train, y_test, evaluate_fn):
    model = train_fn(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_fn(y_test, y_pred)
    return model, metrics

def run_experiment(models, X_train, X_test, y_train, y_test, evaluate_fn):
    results = {}
    trained_models = {}

    for name, train_fn in models.items():
        model = train_fn(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = evaluate_fn(y_test, y_pred)
        trained_models[name] = model
    
    return trained_models, results

