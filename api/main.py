import os
import gdown
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import cv2

# Clases del modelo
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia', 'Lung_Opacity']

# Ruta local del modelo
MODEL_PATH = "modelos/modelo_diagnostico.h5"

# URL para descarga desde Google Drive
GOOGLE_DRIVE_ID = "1ARK-Xj1KiB78McFCqJHJ-dUFz781yfG8"
GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"

# Descarga automática si no existe
if not os.path.exists(MODEL_PATH):
    print("🔽 Descargando modelo desde Google Drive...")
    gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)
    print("✅ Modelo descargado")

# Cargar modelo
print("🧠 Cargando modelo...")
model = load_model(MODEL_PATH)
print("✅ Modelo listo")

# Crear app
app = FastAPI(title="Diagnóstico Médico por Imagen", version="1.0")

# CORS si quieres conectarlo a una web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta principal
@app.get("/")
def root():
    return {"mensaje": "API de Diagnóstico Pulmonar Operativa 🚀"}

# Ruta de predicción
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer la imagen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return {"error": "No se pudo leer la imagen"}

        # Preprocesar
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = img.reshape(1, 224, 224, 1)

        # Predicción
        pred = model.predict(img)[0]
        clase_idx = np.argmax(pred)
        clase = CLASSES[clase_idx]
        prob = float(pred[clase_idx])

        return {
            "archivo": file.filename,
            "prediccion": clase,
            "probabilidad": round(prob * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}

# Para correr localmente: uvicorn api.main:app --reload
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
