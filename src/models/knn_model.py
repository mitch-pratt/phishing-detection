from sklearn.neighbors import KNeighborsClassifier

def build_knn(k=5):
    model = KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean"
    )
    return model

def train_knn(X_train, y_train, k=5):
    model = build_knn(k)
    model.fit(X_train, y_train)
    print("KNN trained successfully")
    return model

def predict(model, X_test):
    return model.predict(X_test)