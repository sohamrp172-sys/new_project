# Telco Customer Churn Prediction - MLOps Project

A complete, production-ready machine learning project for predicting customer churn in the telecom industry. This project implements lightweight MLOps practices with clean architecture, model versioning, data monitoring, and API deployment.

## 📋 Project Overview

This project combines multiple telco churn datasets and implements a complete ML pipeline:

- **Data**: Merges 7 Kaggle datasets into unified training data
- **ML Model**: Binary classification using Logistic Regression & Random Forest
- **Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- **MLOps**: Experiment tracking, model versioning, data monitoring
- **Deployment**: FastAPI REST API for predictions
- **Containerization**: Docker for easy deployment

## 📁 Project Structure

```
telco-churn-mlops/
├── data/
│   ├── raw/                    # Raw CSV datasets
│   └── processed/              # Processed & engineered data
├── models/                     # Trained models & registry
├── src/
│   ├── data/                   # Data merging & preprocessing
│   ├── features/               # Feature engineering
│   ├── models/                 # Training, evaluation, registry
│   ├── api/                    # FastAPI application
│   └── monitoring/             # Data drift detection
├── experiments/                # Metrics, versioning, reports
├── tests/                      # Unit tests
├── train.py                    # Main pipeline orchestrator
├── monitor.py                  # Production monitoring script
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container configuration
├── README.md                   # This file
└── .gitignore                  # Git exclusions
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)

### 1. Setup Local Environment

```bash
# Clone/navigate to project
cd telco-churn-mlops

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Data

Place your CSV files in `data/raw/`:
```bash
data/raw/
├── cell2cellholdout.csv
├── cell2celltrain.csv
├── churn-bigml-20.csv
├── churn-bigml-80.csv
├── customer_churn_1M.csv
├── customer_churn_prediction_dataset.csv
└── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 3. Full Deployment (Recommended)

Execute the complete deployment with training, testing, and API server:

```bash
# Option 1: Python deployment script (recommended)
python deploy.py

# Option 2: Bash script
bash deploy.sh

# Option 3: Windows batch script
deploy.bat
```

This automatically:
1. Installs dependencies
2. Trains models and evaluates
3. Registers the best model
4. Runs test suite
5. Starts data drift monitoring (background)
6. Launches API server on http://localhost:8000

### 4. Train Only (Alternative)

If you only want to train without starting the API:

```bash
python -m src.models.train
```

Output:
- Processed data in `data/processed/`
- Models in `models/`
- Metrics in `experiments/metrics.json`
- Registry in `models/registry.json`
- Training report in `experiments/training_info.json`

### 5. Run API Locally

If API is not already running from deployment script:

```bash
python -m uvicorn src.api.main:app --reload
```

API available at: `http://localhost:8000`

**Interactive docs**: `http://localhost:8000/docs`

### 6. Test Predictions

Once API is running (from deploy.py or uvicorn):

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model-info

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 24,
    "monthlycharges": 65.50,
    "totalcharges": 1574.50
  }'
```

Response:
```json
{
  "churn_probability": 0.35,
  "churn_prediction": 0,
  "prediction_label": "No Churn"
}
```

### 7. Monitor for Data Drift

If not already running from deployment script:

```bash
python monitor.py
```

Checks if new production data shows significant drift from training data.

## 🔧 Configuration

### Data Preprocessing
- **Missing values**: Filled with median (numeric) or mode (categorical)
- **Encoding**: Label encoding for categorical features
- **Scaling**: StandardScaler for numeric features

### Model Configuration
#### Logistic Regression
- max_iter: 1000
- random_state: 42

#### Random Forest
- n_estimators: 100
- max_depth: 15
- min_samples_split: 10
- class_weight: balanced

### Feature Engineering
- Polynomial features (squared)
- Feature interactions
- Log transformations
- Domain-specific features (tenure_binned, customer_lifetime_value)

## 📊 Evaluation Metrics

Models are evaluated on:
- **Accuracy**: Overall correctness
- **Precision**: True positives / predicted positives
- **Recall**: True positives / actual positives
- **F1 Score**: Harmonic mean of precision & recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: TP, TN, FP, FN breakdown

Results saved in `experiments/metrics.json`

## 🏭 MLOps Components

### 1. Data Versioning
- Dataset shape, row count, column count
- Churn distribution
- Missing values count
- Saved in `experiments/data_version.json`

### 2. Experiment Tracking
- Model type and hyperparameters
- Performance metrics
- Training timestamp
- Saved in `experiments/`

### 3. Model Registry
- Versioned model storage (`models/churn_model_v1.pkl`)
- Metadata about each model
- Best model tracking
- Saved in `models/registry.json`

### 4. Data Drift Monitoring
- Baseline statistics from training data
- Drift detection on new data
- Percentage change calculation
- Configurable drift threshold
- Saved in `experiments/drift_report.json`

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t telco-churn-api:latest .
```

### Run Container
```bash
docker run -p 8000:8000 telco-churn-api:latest
```

API available at: `http://localhost:8000`

### Environment Variables
Create `.env` file:
```
MODEL_PATH=models/churn_model_v1.pkl
DRIFT_THRESHOLD=0.1
LOG_LEVEL=info
```

## 📈 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model-info` | GET | Model metadata |
| `/predict` | POST | Get churn prediction |

### Prediction Request Format
```json
{
  "tenure": 24,
  "monthlycharges": 65.50,
  "totalcharges": 1574.50
}
```

### Prediction Response Format
```json
{
  "churn_probability": 0.35,
  "churn_prediction": 0,
  "prediction_label": "No Churn"
}
```

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

## 📝 Logs and Artifacts

- **Metrics**: `experiments/metrics.json`
- **Data Version**: `experiments/data_version.json`
- **Drift Report**: `experiments/drift_report.json`
- **Training Info**: `experiments/training_info.json`
- **Model Registry**: `models/registry.json`

## 🔄 Production Workflow

1. **Train** → `python train.py`
2. **Evaluate** → Check `experiments/metrics.json`
3. **Register** → Model auto-registered in `models/registry.json`
4. **Monitor** → `python monitor.py` for drift detection
5. **Retrain** → When drift detected or metrics degrade

## 🛠️ Customization

### Change Model Type
Edit `src/models/train.py`:
```python
# Add XGBoost model
from xgboost import XGBClassifier

model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)
```

### Adjust Drift Threshold
Edit `monitor.py`:
```python
monitor_production_data(..., drift_threshold=0.15)  # 15% threshold
```

### Add More Features
Edit `src/features/feature_engineering.py`:
```python
# Add custom features
df['custom_feature'] = df['col1'] / df['col2']
```

## 📚 Dependencies

- **Data**: pandas, numpy
- **ML**: scikit-learn
- **Serialization**: joblib
- **API**: fastapi, uvicorn, pydantic
- **Utilities**: python-dotenv, requests

See `requirements.txt` for versions.

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/new-feature`
2. Commit changes: `git commit -m "Add feature"`
3. Push: `git push origin feature/new-feature`
4. Create Pull Request

## 📄 License

MIT License

## 📧 Support

For issues or questions, create a GitHub issue.

## 🎓 Learning Resources

- [MLOps Best Practices](https://mlops.community)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Scikit-learn Guide](https://scikit-learn.org/stable/)
- [Docker Docs](https://docs.docker.com)

---

**Status**: Production-Ready ✓
**Last Updated**: 2026
