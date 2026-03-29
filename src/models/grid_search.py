from sklearn.model_selection import GridSearchCV
from src.models.param_grids import knn_params, rf_params, nn_params
from src.models.knn_model import build_knn
from src.models.random_forest import build_random_forest
from src.models.neural_network import build_neural_network

def optimise_model(model, param_grid, X_train, y_train, scoring="f1", cv=5):
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters:", grid.best_params_)
    print("Best Score:", grid.best_score_)

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