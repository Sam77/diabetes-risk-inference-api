from fastapi.testclient import TestClient

from app import app

SAMPLE = {"pregnancies": 2, "glucose": 148, "blood_pressure": 72, "skin_thickness": 35,
          "insulin": 0, "bmi": 33.6, "diabetes_pedigree": 0.627, "age": 50}


def test_health():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_probability():
    with TestClient(app) as client:
        body = client.post("/predict", json=SAMPLE).json()
        assert 0 <= body["risk_probability"] <= 1
        assert isinstance(body["high_risk"], bool)


def test_batch_predict():
    with TestClient(app) as client:
        assert len(client.post("/predict/batch", json=[SAMPLE, SAMPLE]).json()) == 2


def test_invalid_input_rejected():
    with TestClient(app) as client:
        assert client.post("/predict", json={**SAMPLE, "age": -5}).status_code == 422
