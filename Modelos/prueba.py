import os, io, json, time, requests, base64
from PIL import Image

BASE_URL = "https://ensemble-api-qzpf.onrender.com"
IMAGE_PATH = r"C:\Users\ASUS\OneDrive\Desktop\Viral Pneumonia-109.png" #Coloque la ruta de la imagen para ser analizada

TARGET_SIZE = (224, 224)  # ayudamos al server
JPEG_QUALITY = 85
TIMEOUT = 60
MAX_RETRIES = 3

def pretty(x):
    try: return json.dumps(x, indent=2, ensure_ascii=False)
    except: return str(x)

def prepare_image_buffer(path):
    assert os.path.exists(path), f"No existe: {path}"
    img = Image.open(path).convert("RGB").resize(TARGET_SIZE)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    buf.seek(0)
    return buf

def main():
    # Salud (opcional)
    try:
        h = requests.get(f"{BASE_URL}/health", timeout=10)
        print("Health:", h.status_code, h.text)
    except Exception as e:
        print("No se pudo contactar /health:", e)

    buf = prepare_image_buffer(IMAGE_PATH)
    files = {"file": ("image.jpg", buf, "image/jpeg")}
    data = {"gradcam": "false"}  # sin Grad-CAM

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(f"{BASE_URL}/predict", data=data, files=files, timeout=TIMEOUT)
        except requests.RequestException as e:
            print(f"[Intento {attempt}] Error de red:", e)
            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1)); buf.seek(0); continue
            return

        if r.status_code == 200:
            payload = r.json()
            print("\n=== Resultado ===")
            print("Top class:", payload.get("top_class"))
            print("Confidence:", f"{payload.get('confidence', 0):.4f}")
            print("Margin top2:", f"{payload.get('margin_top2', 0):.4f}")
            print("Uncertain?:", payload.get("uncertain"))
            print("Model disagreement:", f"{payload.get('model_disagreement', 0):.2f}")

            print("\nEnsemble (ordenado):")
            for k, v in sorted((payload.get("ensemble") or {}).items(), key=lambda kv: kv[1], reverse=True):
                print(f"  {k}: {v:.4f}")

            print("\nPor modelo:")
            for m, probs in (payload.get("per_model") or {}).items():
                best = max(probs, key=probs.get)
                print(f"  {m}: {best} ({probs[best]:.4f})")
            return

        if r.status_code == 502 and attempt < MAX_RETRIES:
            print(f"[Intento {attempt}] 502 de Render. Reintentando…")
            time.sleep(2 ** (attempt - 1)); buf.seek(0); continue

        print("Error /predict:", r.status_code)
        try:
            print(pretty(r.json()))
        except Exception:
            print(r.text)
        return

if __name__ == "__main__":
    main()
