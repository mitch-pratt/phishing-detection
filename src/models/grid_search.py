import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from src.models.param_grids import knn_params, rf_params, nn_params
from src.models.knn_model import build_knn
from src.models.random_forest import build_random_forest
from src.models.neural_network import build_neural_network

def optimise_model(model, param_grid, X_train, y_train, scoring="f1", cv=5, top_n=3):
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid.fit(X_train, y_train)
    results = pd.DataFrame(grid.cv_results_)

    from sklearn.model_selection import cross_val_predict

    
    best_model = grid.best_estimator_
    y_pred_cv = cross_val_predict(best_model, X_train, y_train, cv=cv)
    acc = accuracy_score(y_train, y_pred_cv)
    prec = precision_score(y_train, y_pred_cv)
    rec = recall_score(y_train, y_pred_cv)
    f1 = f1_score(y_train, y_pred_cv)

    print(f"\nOptimised Model Metrics:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_

def optimise_knn(X_train, y_train):
    model = build_knn()
    return optimise_model(model, knn_params, X_train, y_train)

def optimise_rf(X_train, y_train):
    model = build_random_forest()
    return optimise_model(model, rf_params, X_train, y_train)

def optimise_nn(X_train, y_train):
    model = build_neural_network()
    return optimise_model(model, nn_params, X_train, y_train)