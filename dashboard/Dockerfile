FROM python:3.13-slim
WORKDIR /app

COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the dashboard + the data it reads
COPY dashboard/ /app/dashboard/
COPY results/ /app/results/
COPY data/processed/zone_centroids.csv /app/data/processed/
COPY data/processed/zone_day_demand.parquet /app/data/processed/

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]   