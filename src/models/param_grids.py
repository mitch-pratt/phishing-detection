knn_params = {
    "n_neighbors": [3, 5, 7, 9],
    "metric": ["euclidean", "manhattan"]
}

rf_params = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

nn_params = {
    "hidden_layer_sizes": [(64, 32), (128, 64), (100,)],
    "activation": ["relu", "tanh"],
    "learning_rate_init": [0.001, 0.01],
    "max_iter": [300, 500]
}