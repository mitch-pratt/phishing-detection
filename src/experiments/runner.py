from src.pipeline.core import train_model, evaluate_model

def run_experiment(models, X_train, X_test, y_train, y_test, evaluate_fn):
    results = {}
    trained_models = {}

    for name, train_fn in models.items():
        print(f"\nTraining {name}...")

        model = train_model(train_fn, X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, evaluate_fn)

        results[name] = metrics
        trained_models[name] = model
    
    return trained_models, results