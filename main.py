from sklearn.model_selection import train_test_split
from src.features.feature_matrix import build_feature_matrix
from src.models.random_forest import train_random_forest, predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.features.feature_extraction import extract_features

import pandas as pd
"""
test_urls = ["http://example.com/login",
             "http://secure.example.com",
             "http:/payp@l.secure.login.com/secure-login"

        ]
for url in test_urls:
    features = extract_features(url)
    print(features)

"""

df = pd.read_csv(
    "data/urlset.csv",
    encoding="latin-1",
    engine="python",
    on_bad_lines="skip"
)

df = df.dropna(subset=["label"])
urls = df["domain"].astype(str).tolist()
labels = df["label"].astype(int).tolist()

"""
print("Number of URLs:\n", len(urls))
print("Number of labels:\n", len(labels))
print("Sample URLs:\n", urls[:3])
print("Sample labels:\n", labels[:3])
"""

X, y = build_feature_matrix(urls, labels)

"""
print("Feature matrix shape:", X.shape)
print("Label vector shape:", y.shape)

print("First feature vector:", X[0])
print("First label:", y[0])

"""

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

model = train_random_forest(X_train, y_train)

print("Random Forest trained successfully")

y_pred = predict(model, X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")