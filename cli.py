#!/usr/bin/env python3
"""
PhishGuard CLI — Command-line URL Scanner
Usage: python cli.py <url> [url2 url3 ...]
       python cli.py --file urls.txt
       python cli.py --interactive
"""

import sys
import os
import json
import argparse
import pickle
from pathlib import Path
from datetime import datetime

# Color codes
RED    = '\033[91m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
WHITE  = '\033[97m'
GRAY   = '\033[90m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from utils.feature_extractor import extract_features, features_to_vector


def load_model():
    model_path = BASE_DIR / 'models' / 'phishing_model.pkl'
    if not model_path.exists():
        print(f"{YELLOW}Model not found. Training now...{RESET}")
        from models.trainer import main as train_main
        train_main()
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['metadata']


def scan_url(url: str, model, metadata) -> dict:
    """Scan a single URL."""
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    features = extract_features(url)
    vector = features_to_vector(features)
    
    pred = model.predict([vector])[0]
    prob = model.predict_proba([vector])[0]
    
    phish_prob = float(prob[1])
    
    if phish_prob >= 0.8:   risk = 'CRITICAL'
    elif phish_prob >= 0.6: risk = 'HIGH'
    elif phish_prob >= 0.4: risk = 'MEDIUM'
    elif phish_prob >= 0.2: risk = 'LOW'
    else:                   risk = 'SAFE'
    
    return {
        'url': url,
        'is_phishing': bool(pred == 1),
        'phishing_prob': round(phish_prob * 100, 1),
        'risk': risk,
        'features': features,
    }


def print_result(result: dict, verbose: bool = False):
    """Print formatted result to terminal."""
    url = result['url']
    is_phish = result['is_phishing']
    prob = result['phishing_prob']
    risk = result['risk']

    # Risk colors
    risk_colors = {
        'CRITICAL': RED, 'HIGH': RED, 'MEDIUM': YELLOW,
        'LOW': GREEN, 'SAFE': GREEN
    }
    color = risk_colors.get(risk, WHITE)

    icon = '⚠️ ' if is_phish else '✅ '
    verdict = f"{RED}PHISHING{RESET}" if is_phish else f"{GREEN}LEGITIMATE{RESET}"

    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"  {icon} {BOLD}{url}{RESET}")
    print(f"{'─' * 60}")
    print(f"  Verdict:     {verdict}")
    print(f"  Risk Level:  {color}{BOLD}{risk}{RESET}")
    print(f"  Phishing %:  {color}{prob}%{RESET}")
    
    if verbose:
        print(f"\n  {GRAY}Key Features:{RESET}")
        checks = [
            ('has_ip_address', 'IP Address'),
            ('has_at_symbol', '@ Symbol'),
            ('is_url_shortener', 'URL Shortener'),
            ('has_suspicious_tld', 'Suspicious TLD'),
            ('uses_https', 'HTTPS'),
            ('phishing_keyword_count', 'Phishing Keywords'),
            ('url_length', 'URL Length'),
            ('subdomain_count', 'Subdomain Count'),
        ]
        f = result['features']
        for key, label in checks:
            val = f.get(key, 0)
            indicator = ''
            if key == 'uses_https':
                indicator = f"{GREEN}✓{RESET}" if val else f"{RED}✗{RESET}"
            elif key in ('has_ip_address', 'has_at_symbol', 'is_url_shortener', 'has_suspicious_tld'):
                indicator = f"{RED}✗ YES{RESET}" if val else f"{GREEN}✓ NO{RESET}"
            else:
                indicator = str(val)
            print(f"    {label:<22} {indicator}")


def print_summary(results: list):
    """Print summary for bulk scan."""
    total = len(results)
    phishing = sum(1 for r in results if r['is_phishing'])
    legit = total - phishing
    
    print(f"\n{BOLD}{'=' * 60}")
    print(f"  SCAN SUMMARY")
    print(f"{'=' * 60}{RESET}")
    print(f"  Total Scanned:   {WHITE}{total}{RESET}")
    print(f"  Phishing Found:  {RED}{phishing}{RESET}")
    print(f"  Legitimate:      {GREEN}{legit}{RESET}")
    print(f"  Phishing Rate:   {round(phishing/total*100,1) if total else 0}%")
    print(f"{BOLD}{'=' * 60}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(
        description='PhishGuard — Phishing URL Scanner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py https://suspicious-login.tk/verify
  python cli.py google.com paypal.com
  python cli.py --file urls.txt --verbose
  python cli.py --interactive
        """
    )
    parser.add_argument('urls', nargs='*', help='URLs to scan')
    parser.add_argument('--file', '-f', help='File containing URLs (one per line)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed feature analysis')
    parser.add_argument('--json', '-j', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    print(f"\n{CYAN}{BOLD}  🛡️  PhishGuard — ML Phishing Detector{RESET}")
    print(f"{GRAY}  Loading model...{RESET}")
    
    model, metadata = load_model()
    model_name = metadata.get('model_name', 'Unknown')
    accuracy = metadata.get('metrics', {}).get('accuracy', 0)
    
    print(f"{GRAY}  Model: {model_name} | Accuracy: {accuracy}%{RESET}")

    # Collect URLs
    urls = list(args.urls)
    
    if args.file:
        try:
            with open(args.file) as f:
                file_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            urls.extend(file_urls)
            print(f"{GRAY}  Loaded {len(file_urls)} URLs from {args.file}{RESET}")
        except FileNotFoundError:
            print(f"{RED}Error: File not found: {args.file}{RESET}")
            sys.exit(1)

    # Interactive mode
    if args.interactive or not urls:
        print(f"\n{CYAN}Interactive Mode — Enter URLs to scan (type 'quit' to exit){RESET}")
        while True:
            try:
                url = input(f"\n{BOLD}URL>{RESET} ").strip()
                if url.lower() in ('quit', 'exit', 'q'):
                    break
                if not url:
                    continue
                result = scan_url(url, model, metadata)
                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print_result(result, verbose=True)
            except KeyboardInterrupt:
                break
        print(f"\n{GRAY}Goodbye!{RESET}\n")
        return

    # Batch scan
    results = []
    for url in urls:
        try:
            result = scan_url(url, model, metadata)
            results.append(result)
            if not args.json:
                print_result(result, verbose=args.verbose)
        except Exception as e:
            print(f"{RED}Error scanning {url}: {e}{RESET}")

    if args.json:
        print(json.dumps(results, indent=2))
    elif len(results) > 1:
        print_summary(results)


if __name__ == '__main__':
    main()
