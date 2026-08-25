$ErrorActionPreference = "Stop"

Write-Host "=========================================="
Write-Host "  Starting Data Ingestion Pipeline...     "
Write-Host "=========================================="

Write-Host "`n[1/5] Downloading OpenI Dataset..."
.\.venv\Scripts\python.exe ingestion/download_openi.py

Write-Host "`n[2/5] Generating Image Embeddings (OpenCLIP)..."
.\.venv\Scripts\python.exe ingestion/embed_images.py

Write-Host "`n[3/5] Generating Report Embeddings (PubMedBERT)..."
.\.venv\Scripts\python.exe ingestion/embed_reports.py

Write-Host "`n[4/5] Generating Literature Embeddings..."
.\.venv\Scripts\python.exe ingestion/embed_literature.py

Write-Host "`n[5/5] Uploading to Qdrant Cloud..."
.\.venv\Scripts\python.exe ingestion/ingest_to_qdrant.py

Write-Host "`n=========================================="
Write-Host "  Data Ingestion Pipeline Complete!       "
Write-Host "=========================================="
