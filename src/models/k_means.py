from sklearn.cluster import KMeans
import numpy as np

def build_kmeans(data, max_k=10):
    inertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    return inertias

def train_kmeans(X_train, y_train):
    model = KMeans(n_clusters=2, random_state=42)
    clusters = model.fit_predict(X_train)

    mapping = {}
    for c in np.unique(clusters):
        labels = y_train[clusters == c]
        mapping[c] = np.bincount(labels).argmax()
    
    model.cluster_label_map = mapping
    print("KMeans trained successfully")
    return model

def predict(model, X_test):
    return model.predict(X_test)