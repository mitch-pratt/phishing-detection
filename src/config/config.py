from src.models.neural_network import train_neural_network
from src.models.knn_model import train_knn
from src.models.k_means import train_kmeans
from src.models.random_forest import train_random_forest

models = {
    "Random Forest": train_random_forest,
    "KNN": train_knn,
    "Neural Network": train_neural_network,
    "K-Means": train_kmeans
}

MODEL_MENU = {
    "1": "Random Forest",
    "2": "KNN",
    "3": "Neural Network",
    "4": "K-Means"
}

feature_names = [
    "URL length",
    "Dot count",
    "Hyphen count",
    "Digit count",
    "@ symbol present",
    "Subdomain count",

    "Suspicious word count",

    "Raw word count",
    "Average word length",
    "Longest word length",
    "Shortest word length",
    "Word length std",
    "Digit ratio",

    "Domain length",
    "Path length",
    "Path level",

    "Dash count in hostname",

    "Special char count",
    "Underscore count",
    "Percent encoding count",
    "Ampersand count",
    "Hash count",

    "Query component count",

    "No HTTPS",
    "IP address in URL",
    "HTTPS token in hostname",

    "Domain keyword in subdomain",

    "Known TLD",

    "Consecutive character repeat",
    "Punycode",

    "Contains WWW",
    "Contains .com",
    "vowel_ratio",
    "digits_in_hostname",
    "longest_subdomain_length",
    "suspicious_tld"
]