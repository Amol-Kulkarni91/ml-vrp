"""Load test: 50 concurrent users, 60 seconds. Run locally against the deployed API."""
from locust import HttpUser, between, task

BASE = "https://ml-vrp.onrender.com/"  # ← replace with your URL

class APIUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(3)
    def forecast(self):
        self.client.get("/forecast")

    @task(3)
    def routes(self):
        self.client.get("/routes")

    @task(2)
    def predict(self):
        self.client.post("/predict", json={
            "lag_1": 120, "lag_7": 95,
            "rolling_mean_7": 110, "rolling_std_7": 15,
            "day_of_week": 2, "is_weekend": 0,
        })

    @task(1)
    def health(self):
        self.client.get("/health")   