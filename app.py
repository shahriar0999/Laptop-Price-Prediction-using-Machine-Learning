import os
import sys
import yaml
from pathlib import Path
from contextlib import asynccontextmanager

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger("fastapi_app")

# ── 1. Load params.yaml (same as your script) ────────────────────────
PARAMS_PATH = ROOT / "params.yaml"

def load_params(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

params = load_params(PARAMS_PATH)

# ── 2. Auth (DagsHub PAT from .env) ─────────────────────────────────
load_dotenv()
token = os.getenv("DAGSHUB_PAT")
if not token:
    raise EnvironmentError("DAGSHUB_PAT is not set. Add it to your .env file.")
os.environ["MLFLOW_TRACKING_USERNAME"] = token
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

# ── 3. MLflow config straight from params ────────────────────────────
TRACKING_URI    = params["mlflow"]["tracking_uri"]
EXPERIMENT_NAME = params["mlflow"]["experiment_train"]

mlflow.set_tracking_uri(TRACKING_URI)


# ── 4. Model loader — exact same logic as your script ────────────────
def load_latest_model():
    """
    Replicates your script exactly:
        runs = mlflow.search_runs(...)
        best_run = runs.sort_values("start_time", ascending=False).iloc[0]
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    """
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found on DagsHub. "
            "Check params.yaml → mlflow.experiment_train"
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )
    if runs.empty:
        raise RuntimeError(
            f"No runs found in experiment '{EXPERIMENT_NAME}'. "
            "Run the training pipeline first."
        )

    # Latest run by start_time — same as your script
    best_run = runs.sort_values("start_time", ascending=False).iloc[0]
    run_id   = best_run.run_id
    model_uri = f"runs:/{run_id}/model"

    logger.info(f"Latest run_id : {run_id}")
    logger.info(f"Loading model : {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)
    logger.info(f"Model loaded ✔  steps: {[s[0] for s in model.steps]}")
    return model, model_uri, run_id


# ── 5. Global model holder ────────────────────────────────────────────
class ModelHolder:
    model     = None
    model_uri = ""
    run_id    = ""


# ── 6. Lifespan: load once at startup ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 55)
    logger.info("  Starting Laptop Price Prediction API")
    logger.info(f"  Tracking URI : {TRACKING_URI}")
    logger.info(f"  Experiment   : {EXPERIMENT_NAME}")
    logger.info("=" * 55)
    try:
        ModelHolder.model, ModelHolder.model_uri, ModelHolder.run_id = load_latest_model()
    except Exception as exc:
        logger.error(f"Startup failed — could not load model: {exc}")
        raise
    yield
    logger.info("Shutting down API …")


# ── 7. FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title="Laptop Price Prediction API",
    description=(
        "Predicts laptop price (₹) via a GradientBoostingRegressor pipeline. "
        "Model loaded from the latest DagsHub MLflow run automatically."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 8. Request / Response schemas ─────────────────────────────────────
class LaptopFeatures(BaseModel):
    Company:         str   = Field(..., example="Dell")
    TypeName:        str   = Field(..., example="Notebook")
    Inches:          float = Field(..., example=15.6)
    display_type:    str   = Field(..., example="Full HD",
                                   description="HD | Full HD | Quad HD | 4K")
    ram:             int   = Field(..., example=8,   description="RAM in GB")
    Weight:          float = Field(..., example=2.0, description="Weight in kg")
    processor_brand: str   = Field(..., example="Intel")
    processor_type:  str   = Field(..., example="Core i5")
    processor_speed: float = Field(..., example=2.5, description="GHz")
    ssd_storage:     int   = Field(0,   example=256, description="SSD in GB")
    hdd_storage:     int   = Field(0,   example=0,   description="HDD in GB")
    flash_storage:   int   = Field(0,   example=0,   description="Flash in GB")
    os:              str   = Field(..., example="Windows")

    model_config = {"json_schema_extra": {"example": {
        "Company": "Dell", "TypeName": "Notebook", "Inches": 15.6,
        "display_type": "Full HD", "ram": 8, "Weight": 2.0,
        "processor_brand": "Intel", "processor_type": "Core i5",
        "processor_speed": 2.5, "ssd_storage": 256,
        "hdd_storage": 0, "flash_storage": 0, "os": "Windows",
    }}}


class PredictionResponse(BaseModel):
    predicted_price_inr: float
    run_id:              str
    model_uri:           str
    status:              str = "success"


class HealthResponse(BaseModel):
    status:       str
    run_id:       str
    model_uri:    str
    model_loaded: bool


# ── 9. Routes ─────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "message":    "Laptop Price Prediction API",
        "docs":       "/docs",
        "health":     "/health",
        "predict":    "/predict  (POST)",
        "model_info": "/model-info",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok" if ModelHolder.model is not None else "model not loaded",
        run_id=ModelHolder.run_id,
        model_uri=ModelHolder.model_uri,
        model_loaded=ModelHolder.model is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: LaptopFeatures):
    if ModelHolder.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    input_df = pd.DataFrame([features.model_dump()])

    try:
        # Pipeline predicts log2(Price) → reverse with 2^x → actual ₹
        log2_price = ModelHolder.model.predict(input_df)[0]
        price_inr  = round(2 ** float(log2_price), 2)
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    logger.info(f"Predicted: ₹{price_inr:,.0f}  (log2={log2_price:.4f})")

    return PredictionResponse(
        predicted_price_inr=price_inr,
        run_id=ModelHolder.run_id,
        model_uri=ModelHolder.model_uri,
    )


@app.get("/model-info", tags=["Model"])
def model_info():
    """Returns info about the currently loaded model."""
    if ModelHolder.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "run_id":          ModelHolder.run_id,
        "model_uri":       ModelHolder.model_uri,
        "experiment":      EXPERIMENT_NAME,
        "tracking_uri":    TRACKING_URI,
        "pipeline_steps":  [s[0] for s in ModelHolder.model.steps],
        "final_estimator": type(ModelHolder.model.steps[-1][1]).__name__,
    }