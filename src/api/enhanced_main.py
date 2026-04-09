"""
Enhanced FastAPI application — all bugs fixed.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
import json
from typing import Optional, List, Dict
from datetime import datetime

from src.models.registry import ModelRegistry
from src.utils.config import config
from src.utils.logger import logger

# ── Globals ────────────────────────────────────────────────────────────────
model            = None
model_n_features = None
feature_names: List[str] = []
shap_explainer   = None
# Training-time threshold for is_high_value (loaded from feature engineer state)
_monthly_charges_p75: float = 75.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_n_features, feature_names, shap_explainer, _monthly_charges_p75

    registry  = ModelRegistry()
    best_path = registry.get_best_model()

    if not best_path or not os.path.exists(best_path):
        logger.warning(f"No best model found at: {best_path}")
    else:
        try:
            model            = joblib.load(best_path)
            model_n_features = model.n_features_in_
            logger.info(f"✓ Model loaded: {best_path} ({model_n_features} features)")

            # ── Load feature names ──────────────────────────────────────
            feature_names_path = os.path.join("experiments", "feature_names.json")
            if os.path.exists(feature_names_path):
                with open(feature_names_path) as f:
                    data = json.load(f)
                feature_names = data.get('features', [])
                logger.info(f"✓ Loaded {len(feature_names)} feature names")

            # FIX: validate feature count matches model expectation
            if feature_names and len(feature_names) != model_n_features:
                logger.error(
                    f"Feature count mismatch: feature_names.json has "
                    f"{len(feature_names)} but model expects {model_n_features}. "
                    "Re-run train_improved.py to regenerate."
                )
                feature_names = []   # force safe fallback

            # ── Load feature engineer state (thresholds) ───────────────
            fe_state_path = os.path.join("experiments", "feature_engineer_state.json")
            if os.path.exists(fe_state_path):
                with open(fe_state_path) as f:
                    fe_state = json.load(f)
                _monthly_charges_p75 = fe_state.get('monthly_charges_p75', 75.0)
                logger.info(f"✓ Loaded feature engineer state (p75={_monthly_charges_p75:.2f})")

            # ── SHAP explainer ──────────────────────────────────────────
            if config.get('api.enable_shap', False) and feature_names:
                try:
                    import shap
                    sample_data_path = os.path.join("data", "processed", "engineered_features.csv")
                    if os.path.exists(sample_data_path):
                        sample_df = pd.read_csv(sample_data_path, nrows=100)
                        if 'churn' in sample_df.columns:
                            sample_df = sample_df.drop(columns=['churn'])

                        # FIX: align sample columns to model's expected features
                        sample_df = sample_df.reindex(columns=feature_names, fill_value=0)

                        # FIX: choose correct explainer based on model type
                        model_type = type(model).__name__
                        tree_models = ('RandomForestClassifier', 'XGBClassifier',
                                       'LGBMClassifier', 'GradientBoostingClassifier')
                        if model_type in tree_models:
                            shap_explainer = shap.TreeExplainer(model, sample_df)
                        else:
                            shap_explainer = shap.KernelExplainer(
                                model.predict_proba, shap.sample(sample_df, 50)
                            )
                        logger.info(f"✓ SHAP explainer initialised ({model_type})")
                except Exception as e:
                    logger.warning(f"Could not initialise SHAP: {e}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")

    yield  # app runs here


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telco Churn Prediction API",
    description="Advanced churn prediction with SHAP explanations",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
_base_dir    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
frontend_dir = os.path.join(_base_dir, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
async def serve_frontend():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. API docs at /docs"}


# ── Schemas ────────────────────────────────────────────────────────────────
class PredictionOutput(BaseModel):
    churn_probability: float
    churn_prediction:  int
    prediction_label:  str
    confidence:        str
    feature_importance: Optional[Dict[str, float]] = None


class BatchPredictionOutput(BaseModel):
    predictions:       List[PredictionOutput]
    total_customers:   int
    predicted_churners: int


class FrontendPredictionInput(BaseModel):
    """Matches exactly what the frontend form sends."""
    tenure:           int   = Field(0, ge=0)
    MonthlyCharges:   float = Field(0.0, ge=0)
    TotalCharges:     float = Field(0.0, ge=0)
    Contract:         str   = "Month-to-month"
    gender:           str   = "Female"
    SeniorCitizen:    int   = Field(0, ge=0, le=1)
    Partner:          str   = "No"
    Dependents:       str   = "No"
    PhoneService:     str   = "Yes"
    MultipleLines:    str   = "No"
    InternetService:  str   = "Fiber optic"
    OnlineSecurity:   str   = "No"
    OnlineBackup:     str   = "No"
    DeviceProtection: str   = "No"
    TechSupport:      str   = "No"
    StreamingTV:      str   = "No"
    StreamingMovies:  str   = "No"
    PaperlessBilling: str   = "Yes"
    PaymentMethod:    str   = "Electronic check"


class SimplePredictionInput(BaseModel):
    tenure:         float = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0)
    totalcharges:   float = Field(..., ge=0)
    contract:       int   = Field(0, ge=0, le=2)
    num_services:   int   = Field(0, ge=0)

    model_config = {"json_schema_extra": {"example": {
        "tenure": 24, "monthlycharges": 65.50, "totalcharges": 1574.50
    }}}


class FullPredictionInput(BaseModel):
    features: Dict[str, float]


class BatchPredictionInput(BaseModel):
    customers: List[Dict[str, float]]


# ── Helpers ────────────────────────────────────────────────────────────────
def _confidence(prob: float) -> str:
    """Return confidence label based on distance from decision boundary."""
    dist = abs(prob - 0.5)
    if dist > 0.35:
        return "high"
    if dist > 0.15:
        return "medium"
    return "low"


def _shap_top_features(features_array: np.ndarray,
                       n: int = 5) -> Optional[Dict[str, float]]:
    """Return top-n features by absolute SHAP value, sorted correctly."""
    if shap_explainer is None or not feature_names:
        return None
    try:
        import shap as shap_lib
        sv = shap_explainer.shap_values(features_array)
        if isinstance(sv, list):
            sv = sv[1]          # binary classification — take positive class
        # FIX: sort by absolute value, take top n
        importance = {feature_names[i]: float(sv[0][i]) for i in range(len(feature_names))}
        sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_imp[:n])
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        return None


def _build_feature_vector(feature_dict: Dict[str, float]) -> np.ndarray:
    """Build a (1, n_features) array in the correct column order."""
    return np.array([[feature_dict.get(f, 0.0) for f in feature_names]])


def _derive_features(feature_dict: Dict[str, float],
                     tenure: float, monthly: float, total: float,
                     contract: int, num_services: int,
                     senior: int = 0) -> Dict[str, float]:
    """Populate all derived features using training-time thresholds."""
    d = feature_dict
    d['tenure']         = tenure
    d['monthlycharges'] = monthly
    d['totalcharges']   = total
    d['contract']       = float(contract)
    d['num_services']   = float(num_services)
    d['senior_citizen'] = float(senior)

    if 'customer_lifetime_value' in d:
        d['customer_lifetime_value'] = monthly * tenure
    if 'avg_monthly_spend' in d:
        d['avg_monthly_spend'] = total / (tenure + 1)
    if 'tenure_to_monthly_ratio' in d:
        d['tenure_to_monthly_ratio'] = tenure / (monthly + 1)
    if 'is_new_customer' in d:
        d['is_new_customer'] = 1.0 if tenure < 6 else 0.0
    if 'is_long_term_customer' in d:
        d['is_long_term_customer'] = 1.0 if tenure > 36 else 0.0
    if 'has_long_contract' in d:
        d['has_long_contract'] = 1.0 if contract > 0 else 0.0
    if 'tenure_squared' in d:
        d['tenure_squared'] = tenure ** 2
    if 'tenure_log' in d:
        d['tenure_log'] = float(np.log1p(max(0, tenure)))
    if 'monthlycharges_log' in d:
        d['monthlycharges_log'] = float(np.log1p(max(0, monthly)))
    if 'totalcharges_log' in d:
        d['totalcharges_log'] = float(np.log1p(max(0, total)))
    # FIX: use training-time percentile, not hardcoded 75
    if 'is_high_value' in d:
        d['is_high_value'] = 1.0 if monthly > _monthly_charges_p75 else 0.0
    if 'service_usage_low' in d:
        d['service_usage_low'] = 1.0 if num_services <= 2 else 0.0
    if 'service_usage_high' in d:
        d['service_usage_high'] = 1.0 if num_services >= 5 else 0.0
    return d


def _predict_and_respond(features: np.ndarray) -> PredictionOutput:
    prediction  = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    return PredictionOutput(
        churn_probability  = probability,
        churn_prediction   = prediction,
        prediction_label   = "Churn" if prediction == 1 else "No Churn",
        confidence         = _confidence(probability),
        feature_importance = _shap_top_features(features),
    )


def _log_prediction(input_data: dict, prediction: int, probability: float):
    log_dir  = os.path.join("logs", "predictions")
    os.makedirs(log_dir, exist_ok=True)
    entry    = {"timestamp": datetime.now().isoformat(),
                "input": input_data, "prediction": prediction,
                "probability": probability}
    log_file = os.path.join(log_dir, f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl")
    with open(log_file, 'a') as f:
        f.write(json.dumps(entry) + '\n')


# ── Endpoints ──────────────────────────────────────────────────────────────

# FIX: health check was missing its decorator
@app.get("/health")
async def health_check():
    return {
        "status":          "healthy",
        "model_loaded":    model is not None,
        "model_n_features": model_n_features,
        "feature_names_ok": len(feature_names) == model_n_features if model_n_features else False,
        "shap_enabled":    shap_explainer is not None,
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict_frontend(input_data: FrontendPredictionInput):
    """Frontend form endpoint — accepts raw string fields."""
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if not feature_names:
        raise HTTPException(503, "Feature names not loaded — re-run train_improved.py")

    try:
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        contract_val = contract_map.get(input_data.Contract, 0)

        yes_fields = [
            input_data.PhoneService, input_data.MultipleLines,
            input_data.OnlineSecurity, input_data.OnlineBackup,
            input_data.DeviceProtection, input_data.TechSupport,
            input_data.StreamingTV, input_data.StreamingMovies,
        ]
        num_services = sum(1 for s in yes_fields if s == "Yes")

        fd = {f: 0.0 for f in feature_names}
        fd = _derive_features(fd,
                              tenure=float(input_data.tenure),
                              monthly=float(input_data.MonthlyCharges),
                              total=float(input_data.TotalCharges),
                              contract=contract_val,
                              num_services=num_services,
                              senior=input_data.SeniorCitizen)

        result = _predict_and_respond(_build_feature_vector(fd))

        if config.get('api.log_predictions', False):
            _log_prediction(input_data.model_dump(), result.churn_prediction,
                            result.churn_probability)
        return result

    except Exception as e:
        logger.error(f"Frontend predict error: {e}")
        raise HTTPException(400, str(e))


@app.post("/predict/simple", response_model=PredictionOutput)
async def predict_simple(input_data: SimplePredictionInput):
    """Simple endpoint — 3-5 core fields, rest imputed."""
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if not feature_names:
        raise HTTPException(503, "Feature names not loaded")

    try:
        fd = {f: 0.0 for f in feature_names}
        fd = _derive_features(fd,
                              tenure=input_data.tenure,
                              monthly=input_data.monthlycharges,
                              total=input_data.totalcharges,
                              contract=input_data.contract,
                              num_services=input_data.num_services)
        return _predict_and_respond(_build_feature_vector(fd))
    except Exception as e:
        logger.error(f"Simple predict error: {e}")
        raise HTTPException(400, str(e))


@app.post("/predict/full", response_model=PredictionOutput)
async def predict_full(input_data: FullPredictionInput):
    """Full endpoint — caller supplies all feature values."""
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if not feature_names:
        raise HTTPException(503, "Feature names not loaded")

    try:
        features = np.array([[input_data.features.get(f, 0.0) for f in feature_names]])
        return _predict_and_respond(features)
    except Exception as e:
        logger.error(f"Full predict error: {e}")
        raise HTTPException(400, str(e))


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(input_data: BatchPredictionInput):
    """Batch endpoint — predict for multiple customers at once."""
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if not feature_names:
        raise HTTPException(503, "Feature names not loaded")

    try:
        preds = []
        for customer in input_data.customers:
            features = np.array([[customer.get(f, 0.0) for f in feature_names]])
            preds.append(_predict_and_respond(features))

        return BatchPredictionOutput(
            predictions        = preds,
            total_customers    = len(preds),
            predicted_churners = sum(1 for p in preds if p.churn_prediction == 1),
        )
    except Exception as e:
        logger.error(f"Batch predict error: {e}")
        raise HTTPException(400, str(e))


@app.get("/model-info")
async def model_info():
    if model is None:
        return {"model_loaded": False}

    registry      = ModelRegistry()
    best_path     = registry.get_best_model()
    reg_info      = registry.get_registry_info()
    best_id       = reg_info.get('best_model')
    best_info     = reg_info.get('models', {}).get(best_id, {})

    return {
        "model_loaded":     True,
        "model_type":       type(model).__name__,
        "model_path":       best_path,
        "n_features":       model_n_features,
        "feature_names":    feature_names,
        "metrics":          best_info.get('metrics', {}),
        "shap_enabled":     shap_explainer is not None,
        "monthly_p75":      _monthly_charges_p75,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,
                host=config.get('api.host', '0.0.0.0'),
                port=config.get('api.port', 8000),
                log_level="info")
