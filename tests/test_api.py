from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    """Ensure the API boots up and the health endpoint responds."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_validation_error():
    """Ensure Pydantic catches bad data (e.g., missing fields)."""
    bad_payload = {
        "ad_spend_cpm": 5.5
        # Missing other required fields
    }
    response = client.post("/predict", json=bad_payload)
    # 422 Unprocessable Entity is the correct FastAPI response for bad data
    assert response.status_code == 422