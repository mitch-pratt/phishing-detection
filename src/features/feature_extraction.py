import re

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

def contains_suspicious_words(url):
    suspicious_words = ["login", "secure", "verify", "update", "account"]
    url_lower = url.lower()
    return 1 if any(word in url_lower for word in suspicious_words) else 0

def extract_features(url):
    return [
        count_dots(url),
        count_hyphens(url),
        url_length(url),
        count_digits(url),
        has_at_symbol(url),
        count_subdomains(url),
        contains_suspicious_words(url)
    ]