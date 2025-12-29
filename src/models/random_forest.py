from sklearn.ensemble import RandomForestClassifier

def build_random_forest():
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    return model

def train_random_forest(X_train, y_train):
    model = build_random_forest()
    model.fit(X_train, y_train)
    print("Random Forest trained successfully")
    return model
    
def predict(model, X_test):
    return model.predict(X_test)

