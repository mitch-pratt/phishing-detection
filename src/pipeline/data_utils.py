from sklearn.model_selection import train_test_split
import numpy as np

from src.config.config import models
from src.dataset.dataset_manager import DataLoader
from src.evaluation.metrics import evaluate_classification
from src.features.feature_extraction import build_feature_matrix

def feature_split(X, y, test_size=0.25, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_and_split():
    
    loader = DataLoader("data/urlset.csv")
    df = loader.load_csv()
    df = loader.clean()
    urls, labels = loader.get_urls_labels()
    
    X, y = build_feature_matrix(urls, labels)
    X_train, X_test, y_train, y_test = feature_split(X, y)

    return X_train, X_test, y_train, y_test

def apply_features(session, features):
    import numpy as np

    features = np.array(features)

    if session.selected_features is None:
        return features

    selected_idx = session.selected_features

    if features.ndim == 1:
        return features[selected_idx]

    elif features.ndim == 2:
        return features[:, selected_idx]

    else:
        raise ValueError(f"Unexpected features shape: {features.shape}")
    
def get_feature_data(session):
    X_train = apply_features(session, session.X_train)
    X_test = apply_features(session, session.X_test)
    return X_train, X_test

def ensure_dataset(session):
    if session.X_train is None:
        print("Building features automatically...")
        session.X_train, session.X_test, session.y_train, session.y_test = build_and_split()

        session.selected_features = list(range(session.X_train.shape[1]))

def initialise_models(session):
 
    if session.X_train is None:
        session.X_train, session.X_test, session.y_train, session.y_test = build_and_split()

    X_train = apply_features(session, session.X_train)
    X_test = apply_features(session, session.X_test)
    y_train = session.y_train
    y_test = session.y_test

    base_models = {
        "Random Forest": models["Random Forest"],
        "KNN": models["KNN"],
        "Neural Network": models["Neural Network"],
        "K-Means": models["K-Means"]
    }

    print("Training base models...")

    for name, train_fn in base_models.items():
        print(f"\nTraining {name}...")
    
        model = train_fn(X_train, y_train)
        
        
        session.models[name] = model

        y_pred = model.predict(X_test)
        metrics = evaluate_classification(y_test, y_pred)
        print(f"{name} trained. Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    print("\nAll base models trained and stored in session.")

