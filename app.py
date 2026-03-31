"""
Phishing Detection Web Application
Flask-based REST API + Web Dashboard
"""

import os
import sys
import json
import time
import pickle
import logging
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

from flask import Flask, render_template, request, jsonify, redirect, url_for

# Add project root to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from utils.feature_extractor import extract_features, features_to_vector, get_feature_names

# ─── Configure Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / 'phishing_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─── Initialize Flask ──────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ─── In-memory Analytics Store ─────────────────────────────────────────────────
analytics = {
    'total_scans': 0,
    'phishing_detected': 0,
    'legitimate_detected': 0,
    'recent_scans': [],  # last 50 scans
}

# ─── Model Loading ─────────────────────────────────────────────────────────────
MODEL_PATH = BASE_DIR / 'models' / 'phishing_model.pkl'
META_PATH  = BASE_DIR / 'models' / 'model_metadata.json'

_model = None
_metadata = None


def get_model():
    global _model, _metadata
    if _model is None:
        if not MODEL_PATH.exists():
            logger.info("Model not found. Training now...")
            _train_model()
        
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
        _model = data['model']
        _metadata = data['metadata']
        logger.info(f"Model loaded: {_metadata.get('model_name', 'Unknown')}")
    
    return _model, _metadata


def _train_model():
    """Train model if not exists."""
    from models.trainer import main as train_main
    train_main()


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Main dashboard."""
    _, metadata = get_model()
    return render_template('index.html', metadata=metadata)


@app.route('/api/scan', methods=['POST'])
def scan_url():
    """Scan a URL for phishing indicators."""
    start_time = time.time()
    
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    
    url = data['url'].strip()
    if not url:
        return jsonify({'error': 'Empty URL'}), 400
    
    # Add scheme if missing
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    try:
        model, metadata = get_model()
        
        # Extract features
        features = extract_features(url)
        feature_vector = features_to_vector(features)
        
        # Predict
        prediction = model.predict([feature_vector])[0]
        probability = model.predict_proba([feature_vector])[0]
        
        phishing_prob = float(probability[1])
        legit_prob = float(probability[0])
        
        is_phishing = bool(prediction == 1)
        confidence = phishing_prob if is_phishing else legit_prob
        
        # Determine risk level
        if phishing_prob >= 0.8:
            risk_level = 'CRITICAL'
            risk_color = '#ff2d55'
        elif phishing_prob >= 0.6:
            risk_level = 'HIGH'
            risk_color = '#ff6b35'
        elif phishing_prob >= 0.4:
            risk_level = 'MEDIUM'
            risk_color = '#ffd60a'
        elif phishing_prob >= 0.2:
            risk_level = 'LOW'
            risk_color = '#34c759'
        else:
            risk_level = 'SAFE'
            risk_color = '#00d4aa'
        
        # Build feature analysis (top contributing features)
        feature_analysis = _analyze_features(features, metadata)
        
        # Parse domain info
        try:
            parsed = urlparse(url)
            domain = parsed.hostname or 'Unknown'
            protocol = parsed.scheme.upper()
        except Exception:
            domain = 'Unknown'
            protocol = 'Unknown'
        
        scan_time = round((time.time() - start_time) * 1000, 2)
        
        result = {
            'url': url,
            'domain': domain,
            'protocol': protocol,
            'is_phishing': is_phishing,
            'phishing_probability': round(phishing_prob * 100, 2),
            'legitimate_probability': round(legit_prob * 100, 2),
            'confidence': round(confidence * 100, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'features': features,
            'feature_analysis': feature_analysis,
            'scan_time_ms': scan_time,
            'timestamp': datetime.now().isoformat(),
            'model_used': metadata.get('model_name', 'Unknown')
        }
        
        # Update analytics
        _update_analytics(result)
        
        logger.info(f"Scanned: {url} → {'PHISHING' if is_phishing else 'LEGITIMATE'} ({phishing_prob:.2%})")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Scan error for {url}: {e}", exc_info=True)
        return jsonify({'error': f'Scan failed: {str(e)}'}), 500


@app.route('/api/bulk-scan', methods=['POST'])
def bulk_scan():
    """Scan multiple URLs at once."""
    data = request.get_json()
    if not data or 'urls' not in data:
        return jsonify({'error': 'No URLs provided'}), 400
    
    urls = data['urls']
    if len(urls) > 20:
        return jsonify({'error': 'Maximum 20 URLs per bulk scan'}), 400
    
    results = []
    for url in urls:
        with app.test_request_context(
            '/api/scan',
            method='POST',
            json={'url': url},
            content_type='application/json'
        ):
            resp = scan_url()
            if hasattr(resp, 'get_json'):
                results.append(resp.get_json())
    
    return jsonify({'results': results, 'total': len(results)})


@app.route('/api/analytics')
def get_analytics():
    """Return current analytics data."""
    _, metadata = get_model()
    
    return jsonify({
        'scan_stats': {
            'total': analytics['total_scans'],
            'phishing': analytics['phishing_detected'],
            'legitimate': analytics['legitimate_detected'],
            'phishing_rate': round(
                analytics['phishing_detected'] / max(analytics['total_scans'], 1) * 100, 1
            )
        },
        'model_info': {
            'name': metadata.get('model_name', 'Unknown'),
            'accuracy': metadata.get('metrics', {}).get('accuracy', 0),
            'f1_score': metadata.get('metrics', {}).get('f1', 0),
            'roc_auc': metadata.get('metrics', {}).get('roc_auc', 0),
            'training_samples': metadata.get('training_samples', 0),
        },
        'recent_scans': analytics['recent_scans'][-10:],
        'all_models': metadata.get('all_models_metrics', {}),
        'feature_importance': metadata.get('feature_importance', [])[:10],
    })


@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Retrain the model."""
    global _model, _metadata
    try:
        logger.info("Starting model retraining...")
        _model = None
        _metadata = None
        _train_model()
        get_model()
        return jsonify({'success': True, 'message': 'Model retrained successfully'})
    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/features/explain')
def explain_features():
    """Return explanation of all features."""
    explanations = {
        'url_length': 'Total character count of the URL. Phishing URLs tend to be longer.',
        'has_ip_address': 'Whether URL uses raw IP instead of domain name.',
        'has_at_symbol': 'Presence of @ in URL, often used to deceive.',
        'has_double_slash_redirect': 'Redirection using // in URL path.',
        'has_dash_in_domain': 'Dashes in domain (e.g., paypal-secure.tk).',
        'subdomain_count': 'Number of subdomains. High counts are suspicious.',
        'dot_count': 'Total dots in URL. More dots may indicate subdomain abuse.',
        'hyphen_count': 'Total hyphens in full URL.',
        'query_param_count': 'Number of query parameters after ?.',
        'path_length': 'Length of URL path component.',
        'domain_length': 'Length of the domain name.',
        'uses_https': 'Whether URL uses HTTPS (encrypted) protocol.',
        'has_url_encoding': 'Presence of percent-encoded characters (%xx).',
        'has_port': 'Non-standard port number in URL.',
        'phishing_keyword_count': 'Count of phishing-related words (login, verify, etc.).',
        'is_url_shortener': 'Whether URL uses a link-shortening service.',
        'has_suspicious_tld': 'Suspicious top-level domain (.tk, .ml, .gq, etc.).',
        'has_hex_characters': 'Hex-encoded characters often used to hide true URL.',
        'digit_count_in_domain': 'Number of digits in domain name.',
        'special_char_count': 'Count of special characters in URL.',
        'has_multiple_tld': 'Multiple TLD-like patterns in URL (deceptive).',
        'url_entropy': 'Shannon entropy (randomness) of URL characters.',
        'domain_digit_ratio': 'Ratio of digits to total characters in domain.',
        'has_brand_name': 'Brand names appearing in suspicious subdomain positions.',
        'domain_in_path': 'Another domain URL appearing in the path.',
        'is_suspicious_pattern': 'Matches known phishing URL patterns.',
    }
    return jsonify(explanations)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _analyze_features(features: dict, metadata: dict) -> list:
    """Return top suspicious and safe features with explanations."""
    analysis = []
    
    suspicious_checks = [
        ('has_ip_address', 'IP address used instead of domain', True),
        ('has_at_symbol', '@ symbol found in URL', True),
        ('is_url_shortener', 'URL shortener service detected', True),
        ('has_suspicious_tld', 'Suspicious top-level domain', True),
        ('has_brand_name', 'Brand name in suspicious position', True),
        ('phishing_keyword_count', 'Phishing keywords found', None),
        ('subdomain_count', 'Excessive subdomains', None),
        ('url_length', 'Unusually long URL', None),
        ('url_entropy', 'High URL randomness/entropy', None),
        ('uses_https', 'No HTTPS encryption', False),
    ]
    
    for feature, description, check_value in suspicious_checks:
        value = features.get(feature, 0)
        is_suspicious = False
        
        if check_value is True:
            is_suspicious = value == 1
        elif check_value is False:
            is_suspicious = value == 0
        elif feature == 'phishing_keyword_count':
            is_suspicious = value > 1
        elif feature == 'subdomain_count':
            is_suspicious = value > 2
        elif feature == 'url_length':
            is_suspicious = value > 100
        elif feature == 'url_entropy':
            is_suspicious = value > 4.5
        
        analysis.append({
            'feature': feature,
            'description': description,
            'value': value,
            'suspicious': is_suspicious,
        })
    
    return analysis


def _update_analytics(result: dict):
    """Update in-memory analytics."""
    analytics['total_scans'] += 1
    if result['is_phishing']:
        analytics['phishing_detected'] += 1
    else:
        analytics['legitimate_detected'] += 1
    
    scan_record = {
        'url': result['url'][:60] + '...' if len(result['url']) > 60 else result['url'],
        'is_phishing': result['is_phishing'],
        'risk_level': result['risk_level'],
        'probability': result['phishing_probability'],
        'timestamp': result['timestamp'],
    }
    
    analytics['recent_scans'].append(scan_record)
    if len(analytics['recent_scans']) > 50:
        analytics['recent_scans'] = analytics['recent_scans'][-50:]


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Pre-load model on startup
    logger.info("Starting Phishing Detector...")
    get_model()
    
    print("\n" + "=" * 55)
    print("  🛡️  PHISHING DETECTOR - Web Dashboard")
    print("=" * 55)
    print("  URL:  http://localhost:5000")
    print("  API:  http://localhost:5000/api/scan")
    print("=" * 55 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
