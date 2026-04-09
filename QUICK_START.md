================================================================================
QUICK START GUIDE - TELCO CHURN MLOPS PROJECT
================================================================================

⚡ 5 MINUTE SETUP

Step 1: Verify You Have Data
  □ Check if data/raw/ has 7 CSV files
  □ If not, copy your telco churn CSVs there

Step 2: Install Dependencies
  python -m venv venv
  venv\Scripts\activate  (Windows)
  source venv/bin/activate  (Linux/Mac)
  pip install -r requirements.txt

Step 3: Train Model
  python train.py

Step 4: Start API
  python -m uvicorn src.api.main:app --reload

Step 5: Test Prediction
  Open: http://localhost:8000/docs
  Or use curl:
  
  curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{
      "tenure": 24,
      "monthlycharges": 65.50,
      "totalcharges": 1574.50
    }'

================================================================================

COMMON COMMANDS

TRAIN THE MODEL:
  python train.py
  
  This will:
    • Merge all 7 datasets
    • Clean and prepare data
    • Create advanced features
    • Train 2 models
    • Evaluate performance
    • Register best model
  
  Time: 2-5 minutes
  Output: models/churn_model_v1.pkl

START API SERVER:
  python -m uvicorn src.api.main:app --reload
  
  Then visit:
    • API: http://localhost:8000/predict
    • Docs: http://localhost:8000/docs
    • Health: http://localhost:8000/health

CHECK RESULTS:
  cat experiments/metrics.json
  
  Shows all model metrics and performance

MONITOR FOR DATA DRIFT:
  python monitor.py
  
  Checks if new data looks different from training

RUN TESTS:
  pytest tests/test_pipeline.py -v

SETUP HELPER:
  python setup.py
  
  Verifies project structure and dependencies

================================================================================

DOCKER DEPLOYMENT

Build:
  docker build -t telco-churn-api:latest .

Run:
  docker run -p 8000:8000 telco-churn-api:latest

Access:
  http://localhost:8000/docs

================================================================================

FILE LOCATIONS

Where to find things:
  • Raw data: data/raw/
  • Processed data: data/processed/
  • Trained model: models/churn_model_v1.pkl
  • Model registry: models/registry.json
  • Performance metrics: experiments/metrics.json
  • Data drift report: experiments/drift_report.json
  • Training details: experiments/training_info.json

================================================================================

API USAGE

Endpoint: POST /predict

Example Request:
  {
    "tenure": 24,
    "monthlycharges": 65.50,
    "totalcharges": 1574.50
  }

Example Response:
  {
    "churn_probability": 0.35,
    "churn_prediction": 0,
    "prediction_label": "No Churn"
  }

Other Endpoints:
  GET /health → {"status": "healthy", "model_loaded": true}
  GET /model-info → {"model_loaded": true, "model_type": "RandomForestClassifier"}

================================================================================

UNDERSTANDING THE METRICS

Accuracy: Percentage of correct predictions (0-100%)
  - Higher is better
  - Good baseline, but not sufficient alone

Precision: Of predicted churns, how many are correct?
  - Important if false alarms are costly
  - Higher = fewer false positives

Recall: Of actual churns, how many did we catch?
  - Important if missing churns is costly
  - Higher = fewer false negatives

F1 Score: Balance between Precision and Recall
  - Good overall metric
  - Useful when you care about both false positives and negatives

ROC-AUC: How well model separates churners from non-churners
  - 0.5 = random guessing
  - 1.0 = perfect separation
  - 0.7-0.8 = good
  - 0.8+ = excellent

Confusion Matrix:
  TP (True Positive):  Correctly predicted churners
  TN (True Negative):  Correctly predicted non-churners
  FP (False Positive): Non-churners predicted as churners
  FN (False Negative): Churners predicted as non-churners

================================================================================

TROUBLESHOOTING

Problem: "No CSV files found in data/raw/"
  Solution: Copy your telco churn datasets to data/raw/

Problem: "ModuleNotFoundError: No module named 'pandas'"
  Solution: pip install -r requirements.txt

Problem: "Model not loaded" in API
  Solution: 
    1. Run: python train.py
    2. Wait for completion
    3. Restart API server

Problem: Port 8000 already in use
  Solution: python -m uvicorn src.api.main:app --port 8001

Problem: "FileNotFoundError: models/churn_model_v1.pkl"
  Solution: You must run training first: python train.py

Problem: Tests failing
  Solution: pip install pytest, then run: pytest tests/test_pipeline.py

================================================================================

NEXT STEPS

After initial setup:

1. REVIEW CODE
   - Read src/data/merge_datasets.py to understand data pipeline
   - Read src/models/train.py to see model training
   - Read src/api/main.py to understand API

2. CUSTOMIZE MODELS
   - Edit src/models/train.py to try different parameters
   - Add new models (XGBoost, Neural Networks, etc.)
   - Experiment with hyperparameters

3. ADD FEATURES
   - Edit src/features/feature_engineering.py
   - Create domain-specific features for your use case
   - Improve model accuracy

4. DEPLOY TO PRODUCTION
   - Build Docker image: docker build -t telco-churn-api .
   - Use cloud platforms (AWS, GCP, Azure)
   - Add authentication if needed
   - Set up monitoring and logging

5. CONTINUOUS IMPROVEMENT
   - Monitor drift: python monitor.py regularly
   - Retrain when accuracy drops
   - Track metrics over time
   - Update models with new data

================================================================================

USEFUL LINKS

Documentation:
  • README.md - Full documentation
  • summary.txt - Project overview
  • PROJECT_STRUCTURE.md - Architecture details

Frameworks:
  • scikit-learn: https://scikit-learn.org
  • FastAPI: https://fastapi.tiangolo.com
  • pandas: https://pandas.pydata.org

MLOps Resources:
  • MLOps.community
  • Made With ML
  • Chip Huyen's MLOps articles

================================================================================

SUPPORT & HELP

If you encounter issues:

1. Read README.md for detailed docs
2. Read summary.txt for architecture explanation
3. Check PROJECT_STRUCTURE.md for file locations
4. Review error messages - they usually indicate the solution
5. Check if dependencies are installed: pip install -r requirements.txt

Common Issues Already Solved:
  ✓ Data loading
  ✓ Preprocessing
  ✓ Model training
  ✓ API deployment
  ✓ Data drift detection
  ✓ Docker containerization

================================================================================

PROGRESS CHECKLIST

Getting Started:
  ☐ Copy project to workspace
  ☐ Navigate to project directory
  ☐ Read this guide (QUICK_START.md)

Setup:
  ☐ Copy data to data/raw/
  ☐ Create virtual environment
  ☐ Install dependencies (pip install -r requirements.txt)

First Run:
  ☐ Train model (python train.py)
  ☐ Check results (cat experiments/metrics.json)
  ☐ Start API (python -m uvicorn src.api.main:app)

Testing:
  ☐ Test health endpoint (curl http://localhost:8000/health)
  ☐ Make prediction (see API USAGE above)
  ☐ Check docs (http://localhost:8000/docs)

Monitoring:
  ☐ Run monitoring (python monitor.py)
  ☐ Check drift report (cat experiments/drift_report.json)

Advanced:
  ☐ Read full README.md
  ☐ Customize hyperparameters
  ☐ Add custom features
  ☐ Build Docker image
  ☐ Deploy to cloud

================================================================================

You're all set! Start with:
  python train.py

Then:
  python -m uvicorn src.api.main:app

Then visit:
  http://localhost:8000/docs

Enjoy! 🚀

================================================================================
