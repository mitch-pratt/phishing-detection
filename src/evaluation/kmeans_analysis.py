def describe_clusters(model, X_train, y_train, feature_names, top_n=10):
    
    import numpy as np

    labels = model.labels_
    centroids = model.cluster_centers_

    for cluster_idx in range(model.n_clusters):
        cluster_mask = labels == cluster_idx
        cluster_size = np.sum(cluster_mask)
        phishing_count = np.sum(y_train[cluster_mask] == 1)
        legit_count = np.sum(y_train[cluster_mask] == 0)

        print(f"\nCluster {cluster_idx}:")
        print(f"Size: {cluster_size} | Phishing: {phishing_count} | Legitimate: {legit_count}")
        print("Top features by centroid value:")

        centroid = centroids[cluster_idx]
        
        top_indices = np.argsort(centroid)[::-1][:top_n]

        for i in top_indices:
            print(f"  {feature_names[i]}: {centroid[i]:.2f}")