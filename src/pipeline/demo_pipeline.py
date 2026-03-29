from src.config.config import feature_names
from src.dataset.dataset_manager import DataLoader
from src.features.feature_extraction import FeatureMatrixChecker, build_feature_matrix, extract_features
from src.pipeline.pipeline import feature_split

def demo_data_loading():
    print("\n Loading dataset...")
    loader = DataLoader("data/urlset.csv")
    df = loader.load_csv()
    print(f"Raw dataset shape: {df.shape}")
    df = loader.clean()
    print(f"Cleaned dataset shape: {df.shape}")
    urls, labels = loader.get_urls_labels()
    print("\n Dataset summary")
    DataLoader.print_summary(urls, labels)

def demo_feature_engineering():
    
    loader = DataLoader("data/urlset.csv")
    df = loader.load_csv()
    df = loader.clean()
    urls, labels = loader.get_urls_labels()
    sample_url = urls[0]
    features = extract_features(sample_url)
    print("\n Feature extraction example")
    print("Sample URL:", sample_url)
    print("Extracted features:", features)
    

    for name, value in zip(feature_names, features):
        print(f"{name}: {value}")

    print("\n Building feature matrix")
    X, y = build_feature_matrix(urls, labels)
    FeatureMatrixChecker.print_summary(X, y)

    print("\n Train/test split")
    X_train, X_test, y_train, y_test = feature_split(X, y)

    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    return X_train, X_test, y_train, y_test

