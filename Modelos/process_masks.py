import os
import cv2
import numpy as np
from tqdm import tqdm

# === RUTAS ===
ROOT_DIR = os.path.join('data', 'Pulmones')
OUT_DIR  = os.path.join('data', 'Pulmones_Mascara')
os.makedirs(OUT_DIR, exist_ok=True)

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def find_matching_mask(masks_dir, stem):
    for ext in IMG_EXTS:
        cand = os.path.join(masks_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    return None

for clase in os.listdir(ROOT_DIR):
    clase_dir = os.path.join(ROOT_DIR, clase)
    if not os.path.isdir(clase_dir):
        continue

    images_dir = os.path.join(clase_dir, 'images')
    masks_dir  = os.path.join(clase_dir, 'masks')

    if not (os.path.isdir(images_dir) and os.path.isdir(masks_dir)):
        print(f"⚠️  Omitiendo '{clase}' (faltan 'images' o 'masks').")
        continue

    out_class = os.path.join(OUT_DIR, clase)
    os.makedirs(out_class, exist_ok=True)

    img_files = [f for f in os.listdir(images_dir)
                 if os.path.splitext(f)[1].lower() in IMG_EXTS]

    for fname in tqdm(img_files, desc=f"Procesando {clase}", unit="img"):
        img_path = os.path.join(images_dir, fname)
        stem, _  = os.path.splitext(fname)

        mask_path = find_matching_mask(masks_dir, stem)
        if mask_path is None:
            print(f"  ❌ No se encontró máscara para {fname}")
            continue

        img  = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"  ❌ Error leyendo {fname}")
            continue

        mask_bin = (mask > 0).astype(np.uint8) * 255
        if img.shape[:2] != mask.shape[:2]:
            mask_bin = cv2.resize(mask_bin, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        masked = cv2.bitwise_and(img, img, mask=mask_bin)
        out_path = os.path.join(out_class, fname)
        ok = cv2.imwrite(out_path, masked)
        if not ok:
            print(f"  ❌ Error guardando {out_path}")

print(f"✅ Proceso completado. Imágenes guardadas en: {OUT_DIR}")
