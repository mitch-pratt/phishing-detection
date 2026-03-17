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

    feature_names = [
    "URL length","Dot count","Hyphen count","Digit count",
    "@ symbol","Subdomains","Suspicious words","Raw word count",
    "Avg word length","Longest word","Shortest word","Word std",
    "Digit ratio","Domain length","Path length","Path level",
    "Dash hostname","Special chars","Underscore count","Percent encoding",
    "Ampersand count","Hash count","Query components","No HTTPS",
    "IP in URL","HTTPS token","Domain keyword","Known TLD",
    "Consecutive repeat","Punycode","Contains WWW","Contains .com"
    ]
    
    for c in np.unique(clusters):
        labels = y_train[clusters == c]
        mapping[c] = np.bincount(labels).argmax()
        print(f"Cluster {c}:")
        print("Size:", len(labels))
        print("Phishing:", np.sum(labels==1))
        print("Legitmate:",np.sum(labels==0))
        centroid = model.cluster_centers_[c]
        print("Centroid features:")
        
        for name,value in zip(feature_names, centroid):
            print(f" {name}:{value:.2f}")
     
    model.cluster_label_map = mapping
    print("KMeans trained successfully")
    return model

def predict(model, X_test):
    return model.predict(X_test)