"""
Dataset Generator & Model Trainer
Generates synthetic training data and trains ML models for phishing detection.
"""

import os
import sys
import json
import random
import pickle
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.feature_extractor import extract_features, features_to_vector, get_feature_names

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ─── Sample URLs for Training ──────────────────────────────────────────────────

LEGITIMATE_URLS = [
    "https://www.google.com/search?q=python",
    "https://github.com/scikit-learn/scikit-learn",
    "https://stackoverflow.com/questions/tagged/python",
    "https://www.wikipedia.org/wiki/Machine_learning",
    "https://www.amazon.com/dp/B08N5LNQCX",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.reddit.com/r/programming",
    "https://docs.python.org/3/library/",
    "https://www.linkedin.com/in/username",
    "https://twitter.com/home",
    "https://www.nytimes.com/section/technology",
    "https://www.bbc.com/news/technology",
    "https://medium.com/tag/machine-learning",
    "https://www.kaggle.com/competitions",
    "https://arxiv.org/abs/2301.00001",
    "https://pytorch.org/tutorials/",
    "https://www.tensorflow.org/learn",
    "https://flask.palletsprojects.com/",
    "https://pandas.pydata.org/docs/",
    "https://numpy.org/doc/stable/",
    "https://www.apple.com/iphone/",
    "https://www.microsoft.com/en-us/",
    "https://www.netflix.com/browse",
    "https://www.spotify.com/us/",
    "https://www.dropbox.com/home",
    "https://mail.google.com/mail/u/0/",
    "https://calendar.google.com/",
    "https://drive.google.com/drive/my-drive",
    "https://www.paypal.com/us/home",
    "https://www.ebay.com/",
]

PHISHING_URLS = [
    "http://paypal-secure-login.tk/account/verify",
    "http://192.168.1.1/login?redirect=paypal.com",
    "http://amazon-account-update.ml/signin",
    "http://www.google.com.phishing-site.xyz/login",
    "http://secure-bank-login.gq/account/confirm",
    "http://apple-id-verification.top/verify-account",
    "http://microsoft-support-alert.tk/windows/update",
    "http://paypal.com.secure-login.cf/signin",
    "http://login-amazon.win/account/suspended",
    "http://facebook-security-alert.ml/confirm",
    "http://bit.ly/3suspicious-link",
    "http://tinyurl.com/phishing-attempt",
    "http://your-account-suspended.tk/login",
    "http://urgent-verify-now.gq/banking/signin",
    "http://free-iphone-winner-2024.xyz/claim",
    "http://netflix-payment-failed.top/update-billing",
    "http://chase-bank-secure.ml/verify-identity",
    "http://wellsfargo-alert.tk/account/locked",
    "http://irs-refund-pending.gq/claim-refund",
    "http://covid-relief-payment.xyz/apply-now",
    "http://ebay-account@login.suspicious.com/signin",
    "http://appleid-security.phish.tk/verify",
    "http://secure-paypal-login-verification.com.tk/",
    "http://123.456.789.012/bank/login.php",
    "http://update-your-account-amazon.ml/billing",
    "http://google-security-alert.xyz/action-required",
    "http://instagram-verify-account.tk/confirm",
    "http://whatsapp-gift-offer.ml/free-premium",
    "http://zoom-meeting-invitation.gq/join-now",
    "http://crypto-investment-returns.xyz/signup",
]


def generate_training_data(n_legitimate: int = 500, n_phishing: int = 500) -> tuple:
    """Generate training dataset by augmenting base URLs."""
    X, y = [], []
    
    # Generate legitimate URL variations
    for i in range(n_legitimate):
        base_url = random.choice(LEGITIMATE_URLS)
        url = _augment_legitimate_url(base_url, i)
        features = extract_features(url)
        X.append(features_to_vector(features))
        y.append(0)
    
    # Generate phishing URL variations
    for i in range(n_phishing):
        base_url = random.choice(PHISHING_URLS)
        url = _augment_phishing_url(base_url, i)
        features = extract_features(url)
        X.append(features_to_vector(features))
        y.append(1)
    
    return np.array(X), np.array(y)


def _augment_legitimate_url(url: str, seed: int) -> str:
    """Create variations of legitimate URLs."""
    random.seed(seed)
    variations = [
        url,
        url + f"?page={random.randint(1, 10)}",
        url + f"&lang=en",
        url.replace('www.', '') if 'www.' in url else url,
        url + f"#{random.choice(['section1', 'content', 'main', 'top'])}",
    ]
    return random.choice(variations)


def _augment_phishing_url(url: str, seed: int) -> str:
    """Create variations of phishing URLs."""
    random.seed(seed)
    suspicious_paths = [
        '/login', '/signin', '/verify', '/confirm', '/update',
        '/secure', '/account', '/billing', '/payment', '/reset-password',
        '/account-verify', '/login.php', '/signin.html', '/secure-login'
    ]
    suspicious_params = [
        '?redirect=paypal.com', '?token=abc123&user=victim',
        '?action=verify&email=user@email.com',
        '?session=expired&update=required',
        '?code=security-alert&click=here'
    ]
    
    variations = [
        url,
        url + random.choice(suspicious_paths),
        url + random.choice(suspicious_params),
        url.replace('http://', 'http://secure-') if 'http://' in url else url,
        url + f"/step{random.randint(1, 3)}",
    ]
    return random.choice(variations)


def train_models(X_train, X_test, y_train, y_test) -> dict:
    """Train multiple ML models and return results."""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42,
                class_weight='balanced'
            ))
        ])
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'model': model,
            'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
            'precision': round(precision_score(y_test, y_pred) * 100, 2),
            'recall': round(recall_score(y_test, y_pred) * 100, 2),
            'f1': round(f1_score(y_test, y_pred) * 100, 2),
            'roc_auc': round(roc_auc_score(y_test, y_prob) * 100, 2),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        print(f"    ✓ Accuracy: {results[name]['accuracy']}%  |  F1: {results[name]['f1']}%  |  AUC: {results[name]['roc_auc']}%")
    
    return results


def get_feature_importance(model, feature_names: list) -> list:
    """Extract feature importance from tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_steps'):
        clf = model.named_steps.get('clf', None)
        if hasattr(clf, 'coef_'):
            importances = abs(clf.coef_[0])
        else:
            return []
    else:
        return []
    
    importance_list = [
        {'feature': name, 'importance': round(float(imp), 4)}
        for name, imp in zip(feature_names, importances)
    ]
    return sorted(importance_list, key=lambda x: x['importance'], reverse=True)


def save_model(model, path: str, metadata: dict):
    """Save model and metadata to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'model': model, 'metadata': metadata}, f)


def load_model(path: str) -> tuple:
    """Load model and metadata from disk."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['metadata']


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("  PHISHING DETECTION - MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    print("\n[1/4] Generating training data...")
    X, y = generate_training_data(n_legitimate=800, n_phishing=800)
    print(f"  ✓ Generated {len(X)} samples ({sum(y==0)} legitimate, {sum(y==1)} phishing)")
    
    print("\n[2/4] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  ✓ Train: {len(X_train)} | Test: {len(X_test)}")
    
    print("\n[3/4] Training models...")
    results = train_models(X_train, X_test, y_train, y_test)
    
    print("\n[4/4] Saving best model...")
    # Pick best model by F1 score
    best_name = max(results, key=lambda k: results[k]['f1'])
    best_result = results[best_name]
    
    feature_names = get_feature_names()
    feature_importance = get_feature_importance(best_result['model'], feature_names)
    
    metadata = {
        'model_name': best_name,
        'feature_names': feature_names,
        'feature_importance': feature_importance,
        'metrics': {k: v for k, v in best_result.items() if k != 'model'},
        'all_models_metrics': {
            name: {k: v for k, v in res.items() if k != 'model'}
            for name, res in results.items()
        },
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'phishing_model.pkl')
    save_model(best_result['model'], model_path, metadata)
    
    # Save metadata as JSON for web access
    meta_path = os.path.join(os.path.dirname(__file__), 'models', 'model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"  ✅ Best Model: {best_name}")
    print(f"  📊 Accuracy:  {best_result['accuracy']}%")
    print(f"  📊 F1 Score:  {best_result['f1']}%")
    print(f"  📊 ROC-AUC:   {best_result['roc_auc']}%")
    print(f"  💾 Saved to: {model_path}")
    print(f"{'=' * 60}\n")
    
    return results, metadata


if __name__ == '__main__':
    main()
