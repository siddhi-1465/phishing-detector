# 🛡️ PhishGuard — ML Phishing Detection System

A complete machine learning–powered phishing URL detection system with a web dashboard and CLI tool.

---

## 📁 Project Structure

```
phishing_detector/
├── app.py                    # Flask web application & REST API
├── cli.py                    # Command-line scanner tool
├── requirements.txt          # Python dependencies
│
├── utils/
│   └── feature_extractor.py  # URL feature extraction (28+ features)
│
├── models/
│   ├── trainer.py            # Dataset generation & ML model training
│   ├── phishing_model.pkl    # Saved trained model (auto-generated)
│   └── model_metadata.json   # Model metrics & metadata (auto-generated)
│
└── templates/
    └── index.html            # Web dashboard UI
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python models/trainer.py
```
This generates 1,600 training samples and trains 3 ML models (Random Forest, Gradient Boosting, Logistic Regression).

### 3. Launch Web Dashboard
```bash
python app.py
```
Open → **http://localhost:5000**

### 4. Use CLI Scanner
```bash
# Scan single URL
python cli.py https://suspicious-login.tk/paypal/verify

# Scan multiple URLs
python cli.py google.com paypal.com example-phish.tk

# Scan from file (one URL per line)
python cli.py --file urls.txt --verbose

# Interactive mode
python cli.py --interactive

# JSON output
python cli.py https://example.com --json
```

---

## 🔬 How It Works

### Feature Extraction (28+ features)
The system extracts URL features across 5 categories:

| Category | Features |
|---|---|
| **Structure** | URL length, dot count, slash count, path length |
| **Domain** | Subdomain count, domain length, digit ratio, TLD |
| **Protocol** | HTTPS usage, port number, URL encoding |
| **Suspicious Patterns** | IP address, @ symbol, phishing keywords, shorteners |
| **Statistical** | Shannon entropy, special char count |

### ML Models
Three models are trained and compared:
- **Random Forest** — Ensemble of decision trees
- **Gradient Boosting** — Sequential error-correcting trees
- **Logistic Regression** — Linear probabilistic classifier

The best model (by F1 score) is automatically selected for deployment.

### Risk Levels
| Level | Phishing Probability |
|---|---|
| 🟢 SAFE | < 20% |
| 🟡 LOW | 20–40% |
| 🟠 MEDIUM | 40–60% |
| 🔴 HIGH | 60–80% |
| 💀 CRITICAL | > 80% |

---

## 🌐 REST API

### Scan a URL
```http
POST /api/scan
Content-Type: application/json

{ "url": "https://paypal-secure-login.tk/verify" }
```

**Response:**
```json
{
  "url": "http://paypal-secure-login.tk/verify",
  "domain": "paypal-secure-login.tk",
  "is_phishing": true,
  "phishing_probability": 94.2,
  "risk_level": "CRITICAL",
  "features": { ... },
  "scan_time_ms": 12.4
}
```

### Get Analytics
```http
GET /api/analytics
```

### Bulk Scan (up to 20 URLs)
```http
POST /api/bulk-scan
Content-Type: application/json

{ "urls": ["url1", "url2", "url3"] }
```

### Retrain Model
```http
POST /api/retrain
```

---

## 📊 Typical Model Performance

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Random Forest | ~95% | ~95% | ~98% |
| Gradient Boosting | ~94% | ~94% | ~97% |
| Logistic Regression | ~88% | ~88% | ~94% |

---

## 🧰 Tech Stack

- **Python 3.8+**
- **Flask** — Web framework & REST API
- **scikit-learn** — Machine learning models
- **NumPy / Pandas** — Data processing
- **Vanilla JS** — Frontend dashboard

---

## 📝 License

MIT License — Free to use, modify, and distribute.
