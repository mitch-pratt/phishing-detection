knn_params = {
    "n_neighbors": [3, 5, 7],
    "metric": ["euclidean", "manhattan"],
    "weights": ["uniform", "distance"]
}

rf_params = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"]
}

nn_params = {
    "hidden_layer_sizes": [(100,), (128, 64)],
    "activation": ["relu"],
    "learning_rate_init": [0.001, 0.01],
    "max_iter": [300]
}