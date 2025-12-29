from src.models.neural_network import train_neural_network
from src.models.knn_model import train_knn
from src.features.feature_split import feature_split
from src.results.metrics import evaluate_classification
from src.utils.data_loader import DataLoader
from src.utils.data_checker import DataChecker
from src.features.feature_matrix import build_feature_matrix, FeatureMatrixChecker
from src.models.random_forest import train_random_forest, predict


loader = DataLoader("data/urlset.csv")
df = loader.load_csv()
loader.clean()
urls, labels = loader.get_urls_labels()

DataChecker.print_summary(urls, labels)

X, y = build_feature_matrix(urls, labels)
FeatureMatrixChecker.print_summary(X, y)

X_train, X_test, y_train, y_test = feature_split(X, y, 0.25, 42)

model = train_neural_network(X_train, y_train)

y_pred = predict(model, X_test)

evaluate_classification(y_test, y_pred)

"""

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)



"""