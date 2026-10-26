from fastapi.testclient import TestClient
from src.serving.api import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_forecast_returns_list():
    r = client.get("/forecast")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_routes_returns_dict():
    r = client.get("/routes")
    assert r.status_code == 200
    assert "status" in r.json()   