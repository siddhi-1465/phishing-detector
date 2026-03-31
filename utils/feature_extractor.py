"""
Feature Extractor for Phishing Detection
Extracts 30+ features from URLs for ML classification
"""

import re
import math
import urllib.parse
from urllib.parse import urlparse
import ipaddress


# Known URL shorteners
URL_SHORTENERS = {
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
    'buff.ly', 'adf.ly', 'bitly.com', 'shorte.st', 'tiny.cc'
}

# Suspicious keywords commonly found in phishing URLs
PHISHING_KEYWORDS = [
    'login', 'signin', 'secure', 'account', 'update', 'banking', 'verify',
    'paypal', 'ebay', 'amazon', 'apple', 'microsoft', 'google', 'facebook',
    'password', 'confirm', 'suspend', 'unusual', 'activity', 'click', 'free',
    'winner', 'prize', 'urgent', 'alert', 'warning', 'limited', 'offer'
]

# Common legitimate TLDs
LEGITIMATE_TLDS = {
    '.com', '.org', '.net', '.edu', '.gov', '.mil', '.int',
    '.co.uk', '.com.au', '.ca', '.de', '.fr', '.jp', '.in'
}

# Suspicious TLDs often used in phishing
SUSPICIOUS_TLDS = {
    '.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.click',
    '.loan', '.win', '.download', '.racing', '.online', '.science'
}


def extract_features(url: str) -> dict:
    """Extract all features from a URL."""
    features = {}

    # --- URL Structure ---
    features['url_length'] = len(url)
    features['has_ip_address'] = _has_ip_address(url)
    features['has_at_symbol'] = 1 if '@' in url else 0
    features['has_double_slash_redirect'] = 1 if '//' in url[7:] else 0
    features['has_dash_in_domain'] = _has_dash_in_domain(url)
    features['subdomain_count'] = _count_subdomains(url)
    features['dot_count'] = url.count('.')
    features['hyphen_count'] = url.count('-')
    features['underscore_count'] = url.count('_')
    features['slash_count'] = url.count('/')
    features['query_param_count'] = _count_query_params(url)
    features['path_length'] = _get_path_length(url)
    features['domain_length'] = _get_domain_length(url)

    # --- Protocol & Encoding ---
    features['uses_https'] = 1 if url.lower().startswith('https') else 0
    features['has_url_encoding'] = 1 if '%' in url else 0
    features['has_port'] = _has_port(url)

    # --- Suspicious Patterns ---
    features['phishing_keyword_count'] = _count_phishing_keywords(url)
    features['is_url_shortener'] = _is_url_shortener(url)
    features['has_suspicious_tld'] = _has_suspicious_tld(url)
    features['has_hex_characters'] = 1 if re.search(r'%[0-9a-fA-F]{2}', url) else 0
    features['digit_count_in_domain'] = _count_digits_in_domain(url)
    features['special_char_count'] = _count_special_chars(url)
    features['has_multiple_tld'] = _has_multiple_tld(url)
    features['url_entropy'] = _calculate_entropy(url)

    # --- Domain Analysis ---
    features['domain_digit_ratio'] = _domain_digit_ratio(url)
    features['has_brand_name'] = _has_brand_name_in_subdomain(url)
    features['domain_in_path'] = _domain_in_path(url)
    features['is_suspicious_pattern'] = _is_suspicious_pattern(url)

    return features


def get_feature_names() -> list:
    """Return ordered list of feature names."""
    return [
        'url_length', 'has_ip_address', 'has_at_symbol', 'has_double_slash_redirect',
        'has_dash_in_domain', 'subdomain_count', 'dot_count', 'hyphen_count',
        'underscore_count', 'slash_count', 'query_param_count', 'path_length',
        'domain_length', 'uses_https', 'has_url_encoding', 'has_port',
        'phishing_keyword_count', 'is_url_shortener', 'has_suspicious_tld',
        'has_hex_characters', 'digit_count_in_domain', 'special_char_count',
        'has_multiple_tld', 'url_entropy', 'domain_digit_ratio', 'has_brand_name',
        'domain_in_path', 'is_suspicious_pattern'
    ]


def features_to_vector(features: dict) -> list:
    """Convert feature dict to ordered list for ML model."""
    return [features.get(name, 0) for name in get_feature_names()]


# ─── Helper Functions ──────────────────────────────────────────────────────────

def _has_ip_address(url: str) -> int:
    """Check if URL uses IP address instead of domain."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ''
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        # Check for IP-like pattern
        if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', url):
            return 1
        return 0


def _has_dash_in_domain(url: str) -> int:
    try:
        parsed = urlparse(url)
        domain = parsed.hostname or ''
        parts = domain.split('.')
        # Check if main domain part (excluding TLD) has dashes
        if len(parts) >= 2 and '-' in parts[-2]:
            return 1
        return 0
    except Exception:
        return 0


def _count_subdomains(url: str) -> int:
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ''
        parts = hostname.split('.')
        # Subtract main domain and TLD
        return max(0, len(parts) - 2)
    except Exception:
        return 0


def _count_query_params(url: str) -> int:
    try:
        parsed = urlparse(url)
        if parsed.query:
            return len(parsed.query.split('&'))
        return 0
    except Exception:
        return 0


def _get_path_length(url: str) -> int:
    try:
        return len(urlparse(url).path)
    except Exception:
        return 0


def _get_domain_length(url: str) -> int:
    try:
        return len(urlparse(url).hostname or '')
    except Exception:
        return 0


def _has_port(url: str) -> int:
    try:
        parsed = urlparse(url)
        if parsed.port and parsed.port not in (80, 443):
            return 1
        return 0
    except Exception:
        return 0


def _count_phishing_keywords(url: str) -> int:
    url_lower = url.lower()
    return sum(1 for kw in PHISHING_KEYWORDS if kw in url_lower)


def _is_url_shortener(url: str) -> int:
    try:
        hostname = urlparse(url).hostname or ''
        return 1 if hostname in URL_SHORTENERS else 0
    except Exception:
        return 0


def _has_suspicious_tld(url: str) -> int:
    try:
        hostname = urlparse(url).hostname or ''
        for tld in SUSPICIOUS_TLDS:
            if hostname.endswith(tld):
                return 1
        return 0
    except Exception:
        return 0


def _count_digits_in_domain(url: str) -> int:
    try:
        hostname = urlparse(url).hostname or ''
        return sum(1 for c in hostname if c.isdigit())
    except Exception:
        return 0


def _count_special_chars(url: str) -> int:
    special = re.findall(r'[!$&\'()*+,;=~`|\\^{}[\]]', url)
    return len(special)


def _has_multiple_tld(url: str) -> int:
    tlds = ['.com', '.org', '.net', '.edu', '.gov']
    count = sum(1 for tld in tlds if tld in url.lower())
    return 1 if count > 1 else 0


def _calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of the URL."""
    if not text:
        return 0.0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    length = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def _domain_digit_ratio(url: str) -> float:
    try:
        hostname = urlparse(url).hostname or ''
        if not hostname:
            return 0.0
        digits = sum(1 for c in hostname if c.isdigit())
        return round(digits / len(hostname), 4)
    except Exception:
        return 0.0


def _has_brand_name_in_subdomain(url: str) -> int:
    """Check if a well-known brand name appears in subdomain (not main domain)."""
    brands = ['paypal', 'ebay', 'amazon', 'apple', 'microsoft', 'google',
              'facebook', 'netflix', 'bank', 'chase', 'wellsfargo', 'citibank']
    try:
        hostname = urlparse(url).hostname or ''
        parts = hostname.split('.')
        # Only look at subdomains, not the main domain
        subdomains = '.'.join(parts[:-2]) if len(parts) > 2 else ''
        for brand in brands:
            if brand in subdomains.lower():
                return 1
        return 0
    except Exception:
        return 0


def _domain_in_path(url: str) -> int:
    """Check if another domain appears in the URL path."""
    try:
        parsed = urlparse(url)
        path = parsed.path
        # Look for domain-like patterns in path
        if re.search(r'(?:https?://|www\.)\S+', path):
            return 1
        return 0
    except Exception:
        return 0


def _is_suspicious_pattern(url: str) -> int:
    """Detect common phishing URL patterns."""
    patterns = [
        r'@',                          # @ symbol in URL
        r'//.*@',                      # Double slash before @
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP address
        r'(?:login|signin|account|secure|update).*\.',  # Login path in domain
        r'[A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+\.',  # Multiple hyphens in domain
    ]
    for pattern in patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return 1
    return 0
