from sklearn.neural_network import MLPClassifier

def build_neural_network():
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42
    )
    return model

def train_neural_network(X_train, y_train):
    model = build_neural_network()
    model.fit(X_train, y_train)
    print("Neural Network trained successfully")
    return model

def predict(model, X_test):
    return model.predict(X_test)