# Streamlit app image. Build from the repository root:
#   docker build -t fishing-effort-app .
#   docker run -p 8501:8501 fishing-effort-app
# then open http://localhost:8501
FROM python:3.12-slim

WORKDIR /app

# exact package versions (made by src/pin_versions.py) so the saved models load correctly
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

# only what the app needs
COPY app/ app/
COPY src/ src/
COPY models/ models/
COPY reports/ reports/
COPY data/processed/fishing_app_sample.parquet data/processed/fishing_app_sample.parquet

EXPOSE 8501
CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
