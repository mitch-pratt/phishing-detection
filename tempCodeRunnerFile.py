loader = DataLoader("data/urlset.csv")
df = loader.load_csv()
loader.clean()
urls, labels = loader.get_urls_labels()

#DataChecker.print_summary(urls, labels)

X, y = build_feature_matrix(urls, labels)
#FeatureMatrixChecker.print_summary(X, y)

X_train, X_test, y_train, y_test = feature_split(X, y, 0.25, 42)

models = {
    "Random Forest": train_random_forest,
    "KNN": train_knn,
    "Neural Network": train_neural_network
}

trained_models, results = run_experiment(
    models,
    X_train, X_test,
    y_train, y_test,
    evaluate_classification
)


plot_accuracy(results)