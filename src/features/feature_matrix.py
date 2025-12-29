from .feature_extraction import extract_features
import numpy as np

class FeatureMatrixChecker:
    @staticmethod
    def print_summary(X, y):
        print("Feature matrix shape:", X.shape)
        print("Label vector shape:", y.shape)
        print("First feature vector:", X[0])
        print("First label:", y[0])

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