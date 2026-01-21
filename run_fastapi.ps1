# Activar el entorno virtual
.venv\Scripts\Activate.ps1

# Abrir el navegador en /docs
Start-Process "http://localhost:8000/docs"

# Ejecutar FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
