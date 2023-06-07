pip install uvicorn
uvicorn apps.fastapi.app:app --host 0.0.0.0 --port 8000 --workers 8 