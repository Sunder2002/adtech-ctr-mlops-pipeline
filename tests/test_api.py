import pytest
from fastapi.testclient import TestClient
from src.api.main import app

def test_health_check():
    """Verify API is alive and ready."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

def test_bid_validation_error():
    """Verify Pydantic schema enforcement."""
    with TestClient(app) as client:
        # Missing 'query' field
        bad_payload = {"email": "test@miq.com"}
        response = client.post("/bid", json=bad_payload)
        assert response.status_code == 422 

def test_root_landing_page():
    """Verify Jinja2 template rendering."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "TechStream" in response.text