"""FastAPI application for the local recruiter demo."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from house_price.model import (
    FEATURE_IMPORTANCE,
    INPUT_FEATURES,
    VALIDATION_METRICS,
    available_neighborhoods,
    get_model,
)
from house_price.schemas import HouseFeatures

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
app = FastAPI(
    title="Ames House Price Prediction",
    description="An educational demo of the model pipeline in the House Price Prediction notebook.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")
templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    neighborhoods = available_neighborhoods()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "neighborhoods": neighborhoods,
            "metrics": VALIDATION_METRICS,
            "feature_importance": FEATURE_IMPORTANCE,
            "input_feature_count": len(INPUT_FEATURES),
            "total_feature_count": 79,
            "dataset_available": bool(neighborhoods),
        },
    )


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "dataset_available": bool(available_neighborhoods()),
        "model_loaded": get_model.cache_info().currsize > 0,
    }


@app.post("/api/predict")
def predict(features: HouseFeatures) -> dict[str, float | str | int]:
    try:
        model = get_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    if features.neighborhood not in model.neighborhoods:
        raise HTTPException(
            status_code=422,
            detail="Choose a neighborhood listed in the Ames dataset.",
        )

    prediction = model.predict(features.model_features())
    return {
        "predicted_price": round(prediction, 2),
        "currency": "USD",
        "input_features_used": len(INPUT_FEATURES),
        "total_model_features": len(model.feature_columns),
        "defaulted_features": len(model.feature_columns) - len(INPUT_FEATURES),
        "model": "Gradient Boosting",
    }
