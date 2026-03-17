from src.utils.data_loader import DataLoader
from src.features.feature_matrix import build_feature_matrix
from src.features.feature_split import feature_split
from src.results.experiments import rf_feature_subset_experiment
from src.models.neural_network import train_neural_network
from src.models.knn_model import train_knn
from src.models.k_means import train_kmeans
from src.features.feature_split import feature_split
from src.results.metrics import plot_confusion_matrix, evaluate_classification
from src.utils.data_loader import DataLoader
from src.utils.data_checker import DataChecker
from src.features.feature_matrix import build_feature_matrix, FeatureMatrixChecker
from src.models.random_forest import train_random_forest, predict
from src.results.plots import plot_metric, plot_metric_select
from src.results.experiments import run_experiment, run_single_model, map_clusters_to_labels
from src.features.feature_extraction import extract_features
import numpy as np
from src.results.plots import plot_kmeans_feature_separation, plot_feature_importance

# Load dataset
loader = DataLoader("data/urlset.csv")
df = loader.load_csv()
df = loader.clean()

urls, labels = loader.get_urls_labels()

# Build feature matrix
X, y = build_feature_matrix(urls, labels)

# Train/test split
X_train, X_test, y_train, y_test = feature_split(X, y)

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=15,
    random_state=42
)

rfecv = RFECV(
    estimator=model,
    step=1,
    cv=5,
    scoring="accuracy"
)

rfecv.fit(X_train, y_train)

print("Optimal number of features:", rfecv.n_features_)