param([string]$Command = "help")

switch ($Command) {
    "setup"      { pip install -r requirements.txt }
    "ingest"     { python -m src.ingestion.ingest --dir data/sample_docs/ }
    "api"        { uvicorn src.api.main:app --reload --port 8000 }
    "ui"         { streamlit run ui/app.py }
    "eval"       { python -m src.evaluation.benchmark }
    "test"       { pytest tests/ -v }
    "docker-up"  { docker compose up --build -d }
    "docker-down"{ docker compose down }
    "docker-logs"{ docker compose logs -f }
    "logs-api"   { docker logs documind-api-cont -f }
    "logs-ui"    { docker logs documind-ui-cont -f }
    "logs-qdrant"{ docker logs documind-qdrant-cont -f }
    "status"     { docker inspect --format "{{.Name}} → {{.State.Status}}" documind-qdrant-cont documind-api-cont documind-ui-cont }
    "clean" {
        Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force
        Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
        Write-Host "Cleaned."
    }
    default {
        Write-Host "Commands: setup, ingest, api, ui, eval, test, docker-up, docker-down, docker-logs, logs-api, logs-ui, logs-qdrant, status, clean"
    }
}