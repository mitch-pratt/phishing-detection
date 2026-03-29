import re
import statistics
from src.config.config import feature_names
import numpy as np

def tokenize_url(url):
    tokens = re.split(r"[./?=&_\-@]", url.lower())
    return [t for t in tokens if t]

def raw_word_count(url):
    return len(tokenize_url(url))

def avg_word_length(url):
    tokens = tokenize_url(url)
    if not tokens:
        return 0
    return sum(len(t) for t in tokens) / len(tokens)

def longest_word_length(url):
    tokens = tokenize_url(url)
    return max ((len(t) for t in tokens), default=0)

def shortest_word_length(url):
    tokens = tokenize_url(url)
    return min ((len(t) for t in tokens), default=0)

def word_length_std(url):
    tokens = tokenize_url(url)
    if len(tokens) < 2:
        return 0
    return statistics.stdev(len(t) for t in tokens)

def domain_length(url):
    host = get_host(url)
    return len(host)

def path_length(url):
    if "/" not in url:
        return 0
    return len(url.split("/", 1)[1])

KNOWN_TLDS = ["com", "org", "net", "edu", "gov", "co", "uk", "de"]

def has_known_tld(url):
    host = get_host(url)
    parts = host.split(".")
    return 1 if parts[-1] in KNOWN_TLDS else 0

SPECIAL_CHARS = ['-', '.', '/', '@', '?', '&', '=', '_']

def count_special_chars(url):
    return sum(url.count(c) for c in SPECIAL_CHARS)

def consecutive_char_repeat(url):
    return 1 if re.search(r"(.)\1{2,}", url) else 0

def has_punycode(url):
    return 1 if "xn--" in url else 0

def has_www(url):
    return 1 if "www" in url.lower() else 0

def has_com(url):
    return 1 if ".com" in url.lower() else 0

def url_length(url):
    return len(url)

def count_dots(url):
    return url.count('.')

def count_hyphens(url):
    return url.count("-")

def count_digits(url):
    return sum(c.isdigit() for c in url)

def has_at_symbol(url):
    return 1 if "@" in url else 0

# count subdomains #
def get_host(url):
    if "://" in url:
        url = url.split("://")[1]
    return url.split("/")[0]

def count_subdomains(url):
    host = get_host(url)
    parts = host.split(".")
    
    # heuristic #
    if len(parts) <= 2:
        return 0
    return len(parts) - 2
# count subdomains #

def path_level(url):
    if "://" in url:
        url = url.split("://")[1]

    parts = url.split("/",1)

    if len(parts) < 2:
        return 0

    return parts[1].count("/")

def num_dash_hostname(url):
    host = get_host(url)
    return host.count("-")

def count_underscore(url):
    return url.count("_")

def count_percent(url):
    return url.count("%")

def num_query_components(url):
    if "?" not in url:
        return 0
    query = url.split("?")[1]
    return query.count("&") + 1

def count_ampersand(url):
    return url.count("&")

def count_hash(url):
    return url.count("#")

def no_https(url):
    return 1 if not url.lower().startswith("https") else 0

def has_ip_address(url):
    host = get_host(url)
    return 1 if re.match(r"\d+\.\d+\.\d+\.\d+", host) else 0

def https_in_hostname(url):
    host = get_host(url)
    return 1 if "https" in host else 0

def domain_in_subdomain(url):
    host = get_host(url)
    parts = host.split(".")
    if len(parts) <= 2:
        return 0
    
    subdomains = parts[:-2]
    tlds = KNOWN_TLDS

    return 1 if any(s in tlds for s in subdomains) else 0

def digit_ratio(url):
    if len(url) == 0:
        return 0
    return sum(c.isdigit() for c in url) / len(url)


SUSPICIOUS_WORDS = [
    "login","secure","verify","update","account","verification",
    "store","remove","customer","configuration","recover","support",
    "activity","billing","online","provider","protect","services",
    "service","resolved","home","setup","center","summary",
    "contact","server","solution", "access"
]

def contains_suspicious_words(url):
    url_lower = url.lower()
    return sum(word in url_lower for word in SUSPICIOUS_WORDS)

#later - can analyse each subdomain to find most common words/tokens associated with phishing and self update? 

def extract_features(url):
    return [
        # basic url features
        url_length(url),
        count_dots(url),
        count_hyphens(url),
        count_digits(url),
        has_at_symbol(url),
        count_subdomains(url),

        # suspicious words
        contains_suspicious_words(url),

        # word statistics
        raw_word_count(url),
        avg_word_length(url),
        longest_word_length(url),
        shortest_word_length(url),
        word_length_std(url),
        digit_ratio(url),

        # domain/path
        domain_length(url),
        path_length(url),
        path_level(url),

        # hostname structure
        num_dash_hostname(url),

        # character statistics
        count_special_chars(url),
        count_underscore(url),
        count_percent(url),
        count_ampersand(url),
        count_hash(url),

        # query
        num_query_components(url),

        # security indicators
        no_https(url),
        has_ip_address(url),
        https_in_hostname(url),

        # domain tricks
        domain_in_subdomain(url),

        # misc
        has_known_tld(url),
        consecutive_char_repeat(url),
        has_punycode(url),
        has_www(url),
        has_com(url)
    ]

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

from src.config.config import feature_names

def show_selected_features(session):
    if session.selected_features is None:
        
        selected_idx = list(range(len(feature_names)))
    else:
        selected_idx = session.selected_features

    print(f"Active features ({len(selected_idx)}):")
    for i in selected_idx:
        print("-", feature_names[i])