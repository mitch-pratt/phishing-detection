from .feature_extraction import extract_features
import numpy as np

def build_feature_matrix(urls, labels):

    X = []
    y = []

    for url, label in zip(urls, labels):
        features = extract_features(url)
        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y