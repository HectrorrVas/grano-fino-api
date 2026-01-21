from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
from huggingface_hub import hf_hub_download
import io
import os
from collections import Counter

# -------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------

APP_NAME = "GranoFino API"
APP_VERSION = "1.0.0"

# Descargar modelo desde Hugging Face
MODEL_PATH = hf_hub_download(repo_id="F1933/GranoFino", filename="best12.pt")

CONF_THRESHOLD = 0.4
IMG_SIZE = 416

CLASS_NAMES = {
    0: "GBF",
    1: "GIF",
    2: "GSF"
}

# -------------------------------------------------
# INICIALIZACIÓN
# -------------------------------------------------

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "API para la detección del grado de fermentación de granos de cacao "
        "mediante YOLO. "
        "El endpoint /predict/image devuelve una imagen anotada (PNG) y "
        "/predict/json devuelve los resultados en formato JSON."
    )
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo YOLO
model = YOLO(MODEL_PATH)
model.fuse()
model.verbose = False

# -------------------------------------------------
# ENDPOINTS
# -------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "GranoFino API funcionando"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


# -------------------------------------------------
# 1️⃣ ENDPOINT: PREDICCIÓN CON IMAGEN ANOTADA
# -------------------------------------------------

@app.post(
    "/predict/image",
    tags=["Producción"],
    summary="Visual: Devuelve la imagen con cuadros dibujados",
    response_class=StreamingResponse
)
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(image, conf=CONF_THRESHOLD, imgsz=IMG_SIZE)
    r = results[0]

    # Generar imagen anotada en memoria (con etiquetas y porcentajes de confianza)
    annotated = r.plot(labels=True, conf=True)
    annotated = Image.fromarray(annotated[..., ::-1])

    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")


# -------------------------------------------------
# 2️⃣ ENDPOINT: PREDICCIÓN JSON (Para el Despliegue/Interfaz)
# -------------------------------------------------

@app.post(
    "/predict/json",
    tags=["Producción"],
    summary="Datos: Devuelve coordenadas y conteos en JSON",
    description="Ideal para que la interfaz dibuje los cuadros dinámicamente."
)
async def predict_json(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size # <--- Importante para el Frontend

    results = model(image, conf=CONF_THRESHOLD, imgsz=IMG_SIZE)
    r = results[0]

    detections = []
    for box in r.boxes:
        class_id = int(box.cls)
        detections.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, "Desconocido"),
            "confidence": round(float(box.conf), 4),
            "bbox": [round(v, 2) for v in box.xyxy[0].tolist()] # [x1, y1, x2, y2]
        })

    # Calcular estadísticas detalladas
    total_detections = len(detections)
    summary_stats = {}
    
    if total_detections > 0:
        counts = Counter([d["class_name"] for d in detections])
        avg_confidence = sum([d["confidence"] for d in detections]) / total_detections
        
        for class_name, count in counts.items():
            summary_stats[class_name] = {
                "count": count,
                "percentage": f"{round((count / total_detections) * 100, 2)}%"
            }
    else:
        avg_confidence = 0

    print(f"--- Procesada: {file.filename} | Encontrados: {total_detections} granos ---")

    return {
        "info": {
            "filename": file.filename,
            "image_size": {"width": width, "height": height},
            "model": "YOLO11l-GranoFino",
            "average_confidence": f"{round(avg_confidence * 100, 2)}%"
        },
        "summary": summary_stats,
        "total": total_detections,
        "detections": detections
    }
