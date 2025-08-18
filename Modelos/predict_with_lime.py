import os, io, json, time, requests
import numpy as np
from PIL import Image
from lime import lime_image
from skimage.segmentation import slic
from skimage.color import label2rgb

# ========= Config =========
BASE_URL     = "https://ensemble-api-qzpf.onrender.com"
IMAGE_PATH   = r"C:\Users\ASUS\OneDrive\Desktop\R.jpg"  #Coloque en esta parte la ruta de la imagen

TARGET_SIZE  = (224, 224)   # tamaño que espera tu servidor
JPEG_QUALITY = 85
TIMEOUT      = 60
MAX_RETRIES  = 3

# Salidas
OUT_JSON     = "prediccion.json"
OUT_LIME_PNG = "lime_explicacion.png"

def pretty(x):
    try: return json.dumps(x, indent=2, ensure_ascii=False)
    except: return str(x)

def pil_to_jpeg_buffer(pil_im):
    buf = io.BytesIO()
    pil_im.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    buf.seek(0)
    return buf

def open_and_resize(path):
    assert os.path.exists(path), f"No existe: {path}"
    return Image.open(path).convert("RGB").resize(TARGET_SIZE)

def call_predict(pil_im, gradcam=False):
    """
    Llama /predict con una imagen PIL y devuelve el payload JSON.
    """
    buf = pil_to_jpeg_buffer(pil_im)
    files = {"file": ("image.jpg", buf, "image/jpeg")}
    data  = {"gradcam": "true" if gradcam else "false"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(f"{BASE_URL}/predict", data=data, files=files, timeout=TIMEOUT)
        except requests.RequestException as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))
                buf.seek(0)
                continue
            raise RuntimeError(f"Error de red tras {MAX_RETRIES} intentos: {e}")

        if r.status_code == 200:
            return r.json()

        # Render a veces da 502 en cold start
        if r.status_code == 502 and attempt < MAX_RETRIES:
            time.sleep(2 ** (attempt - 1))
            buf.seek(0)
            continue

        # Otro error
        try:
            raise RuntimeError(f"/predict {r.status_code}: {pretty(r.json())}")
        except Exception:
            raise RuntimeError(f"/predict {r.status_code}: {r.text}")

def build_class_order(example_payload):
    """
    LIME necesita que predict_proba devuelva probabilidades en un orden fijo.
    Tomamos el dict 'ensemble' del payload de ejemplo para fijar ese orden.
    """
    ens = example_payload.get("ensemble") or {}
    if not ens:
        raise ValueError("El payload no incluye 'ensemble' con probabilidades por clase.")
    # Ordenaremos por nombre de clase para consistencia reproducible
    return sorted(list(ens.keys()))

def predict_proba_for_lime(np_imgs, class_order):
    """
    Función 'predict_proba' para LIME-Image.
    Recibe una lista/array de imágenes (H,W,3) en rango [0,1] y
    devuelve un array (N, C) con probabilidades según 'class_order'.
    """
    probs = []
    for arr in np_imgs:
        # array -> PIL -> resize (por si LIME perturba y cambia tamaño)
        pil_im = Image.fromarray((arr * 255).astype(np.uint8)).resize(TARGET_SIZE)
        payload = call_predict(pil_im, gradcam=False)
        ens = payload.get("ensemble") or {}
        # Mapear en el orden de clases
        row = [float(ens.get(c, 0.0)) for c in class_order]
        probs.append(row)
    return np.array(probs, dtype=np.float32)

def main():
    # (Opcional) /health
    try:
        h = requests.get(f"{BASE_URL}/health", timeout=10)
        print("Health:", h.status_code, h.text)
    except Exception as e:
        print("No se pudo contactar /health:", e)

    # Imagen base y predicción inicial
    pil_im = open_and_resize(IMAGE_PATH)
    base_payload = call_predict(pil_im, gradcam=False)
    print("\n=== Resultado base ===")
    print(pretty(base_payload))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        f.write(pretty(base_payload))
    print(f"💾 Guardado JSON: {OUT_JSON}")

    # Orden de clases a partir del payload
    class_order = build_class_order(base_payload)
    print("Orden de clases:", class_order)

    # Preparar explainer de LIME
    explainer = lime_image.LimeImageExplainer()

    # LIME espera imagen en [0,1] float32
    np_img = np.asarray(pil_im).astype(np.float32) / 255.0

    # Función predict_proba que consulta a tu API
    def predict_fn(batch):
        return predict_proba_for_lime(batch, class_order)

    # Clase objetivo: la top_class del ensemble
    top_class_name = base_payload.get("top_class")
    if top_class_name not in class_order:
        # como fallback, tomar argmax del ensemble
        top_class_name = class_order[int(np.argmax([base_payload["ensemble"].get(c,0) for c in class_order]))]
    class_idx = class_order.index(top_class_name)
    print(f"Explicando clase objetivo: {top_class_name} (idx={class_idx})")

    # Segmentation para superpixeles (puedes afinar n_segments/compactness)
    segmentation_fn = lambda x: slic(x, n_segments=200, compactness=10, sigma=1, start_label=0)

    # Explicar
    explanation = explainer.explain_instance(
        np_img,
        classifier_fn=predict_fn,
        top_labels=1,                 # solo la clase objetivo
        hide_color=0,
        num_samples=1000,             # más samples => más estable (trade-off tiempo)
        segmentation_fn=segmentation_fn
    )

    # Obtener máscara positiva para la clase objetivo
    temp, mask = explanation.get_image_and_mask(
        label=class_idx,
        positive_only=True,
        num_features=10,              # muestra las 10 regiones más influyentes
        hide_rest=False
    )

    # Guardar overlay bonito
    overlay = label2rgb(mask, temp, bg_label=0, alpha=0.4)
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(overlay).save(OUT_LIME_PNG, format="PNG", optimize=True)
    print(f"✅ LIME guardado en: {OUT_LIME_PNG}")

if __name__ == "__main__":
    main()
