from src.utils.data_loader import DataLoader
from src.features.feature_matrix import build_feature_matrix
from src.features.feature_split import feature_split
from src.results.experiments import random_forest_tree_experiment


# Load dataset
loader = DataLoader("data/urlset.csv")
df = loader.load_csv()
df = loader.clean()

urls, labels = loader.get_urls_labels()

# Build feature matrix
X, y = build_feature_matrix(urls, labels)

# Train/test split
X_train, X_test, y_train, y_test = feature_split(X, y)


# Experiment parameters
tree_values = [10, 50, 100, 200]

# Run experiment
rf_results = random_forest_tree_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    tree_values
)

print("\nExperiment Results:")
for r in rf_results:
    print(r)

import pandas as pd

df_results = pd.DataFrame(rf_results)
df_results.to_csv("rf_tree_experiment.csv", index=False)

print("Results saved to rf_tree_experiment.csv")