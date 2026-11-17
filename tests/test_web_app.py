"""Smoke and integration tests for the local house price demo."""

from fastapi.testclient import TestClient

from house_price.main import app

client = TestClient(app)

SAMPLE_INPUT = {
    "GrLivArea": 1464,
    "LotArea": 9479,
    "OverallQual": 6,
    "YearBuilt": 1973,
    "BedroomAbvGr": 3,
    "FullBath": 2,
    "GarageCars": 2,
    "Neighborhood": "NAmes",
}


def test_home_page_explains_model_and_dataset() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Ames, Iowa" in response.text
    assert "1,460" in response.text
    assert "$16,946" in response.text
    assert "0.9002" in response.text
    assert "training split" in response.text


def test_health_endpoint_reports_dataset() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["dataset_available"] is True


def test_prediction_uses_pipeline_and_training_defaults() -> None:
    response = client.post("/api/predict", json=SAMPLE_INPUT)

    assert response.status_code == 200
    result = response.json()
    assert result["model"] == "Gradient Boosting"
    assert result["currency"] == "USD"
    assert result["input_features_used"] == 8
    assert result["total_model_features"] == 79
    assert result["defaulted_features"] == 71
    assert result["predicted_price"] > 0

    repeat = client.post("/api/predict", json=SAMPLE_INPUT)
    assert repeat.status_code == 200
    assert repeat.json()["predicted_price"] == result["predicted_price"]


def test_prediction_rejects_values_outside_dataset_range() -> None:
    invalid_input = {**SAMPLE_INPUT, "YearBuilt": 2026}
    response = client.post("/api/predict", json=invalid_input)

    assert response.status_code == 422
