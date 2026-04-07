import pytest
from fastapi.testclient import TestClient
from src.api.main import app

# Senior Fix: Use the client as a context manager to trigger 'lifespan' events
def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

def test_bid_validation_error():
    with TestClient(app) as client:
        bad_payload = {"email": "test@miq.com"}
        response = client.post("/bid", json=bad_payload)
        assert response.status_code == 422 

def test_root_landing_page():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "TechStream" in response.text