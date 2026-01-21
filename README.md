# 🍫 GranoFino API - Detección de Fermentación de Cacao

Esta es una API profesional desarrollada con **FastAPI** y **YOLO11** para automatizar la clasificación del grado de fermentación en granos de cacao mediante vision artificial.

## 🚀 Características
- **Detección en tiempo real**: Identifica granos Individualmente.
- **Clasificación**: Separa los granos en tres categorías clave:
  - **GBF**: Grano Bien Fermentado.
  - **GIF**: Grano Insuficientemente Fermentado.
  - **GSF**: Grano Sobre Fermentado / Seco.
- **Doble Endpoint**:
  - `POST /predict/image`: Devuelve una imagen anotada con cuadros y porcentajes de confianza.
  - `POST /predict/json`: Devuelve datos estadísticos y coordenadas exactas para integración con apps.
- **Optimizado para Nube**: El modelo se carga automáticamente desde **Hugging Face Hub**.

## 🛠️ Tecnologías
- **Backend**: FastAPI (Python)
- **IA**: YOLO11 (Ultralytics)
- **Model Storage**: Hugging Face Hub
- **Deployment**: Render (CPU Optimized)

## 📋 Requisitos Locales
Si deseas ejecutar este proyecto localmente:
1. Clonar el repositorio.
2. Crear un entorno virtual: `python -m venv .venv`
3. Activar el entorno y ejecutar: `pip install -r requirements.txt`
4. Lanzar la API: `uvicorn main:app --reload`

## 📡 Endpoints Principales
- **GET /**: Estado de la API.
- **POST /predict/image**: Envía una imagen (multipart/form-data) y recibe un PNG procesado.
- **POST /predict/json**: Envía una imagen y recibe un reporte detallado en formato JSON.

## 👥 Autor
**Héctor Vas**
- Repositorio del Modelo: [F1933/GranoFino](https://huggingface.co/F1933/GranoFino)
