import os
import cv2
import numpy as np
from tqdm import tqdm

# === CAMBIA ESTA RUTA A TU CARPETA ORIGINAL ===
root_dir = r"C:\Users\ASUS\OneDrive\Desktop\Pulmones"
# === CARPETA DONDE SE GUARDARÁN LAS IMÁGENES CON MÁSCARA ===
out_dir = r"C:\Users\ASUS\OneDrive\Desktop\Pulmones_Mascara"

# Crear carpeta de salida
os.makedirs(out_dir, exist_ok=True)

# Extensiones de imagen aceptadas
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def find_matching_mask(masks_dir, stem):
    """Busca una máscara con el mismo nombre base y extensión de imagen válida."""
    for ext in IMG_EXTS:
        mask_candidate = os.path.join(masks_dir, stem + ext)
        if os.path.exists(mask_candidate):
            return mask_candidate
    return None

# Recorrer las clases (COVID, Lung_Opacity, Normal, Viral Pneumonia, etc.)
for clase in os.listdir(root_dir):
    clase_dir = os.path.join(root_dir, clase)
    if not os.path.isdir(clase_dir):
        continue

    images_dir = os.path.join(clase_dir, "images")
    masks_dir  = os.path.join(clase_dir, "masks")

    if not (os.path.isdir(images_dir) and os.path.isdir(masks_dir)):
        print(f"⚠️  Omitiendo '{clase}' (faltan 'images' o 'masks').")
        continue

    # Carpeta de salida para esta clase
    out_class = os.path.join(out_dir, clase)
    os.makedirs(out_class, exist_ok=True)

    # Lista de imágenes válidas
    img_files = [f for f in os.listdir(images_dir)
                 if os.path.splitext(f)[1].lower() in IMG_EXTS]

    # Procesar imágenes
    for fname in tqdm(img_files, desc=f"Procesando {clase}", unit="img"):
        img_path = os.path.join(images_dir, fname)
        stem, _ = os.path.splitext(fname)

        mask_path = find_matching_mask(masks_dir, stem)
        if mask_path is None:
            print(f"  ❌ No se encontró máscara para {fname}")
            continue

        # Leer imagen y máscara
        img  = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"  ❌ Error leyendo {fname}")
            continue

        # Binarizar máscara
        mask_bin = (mask > 0).astype(np.uint8)

        # Asegurar mismo tamaño
        if img.shape[:2] != mask.shape[:2]:
            mask_bin = cv2.resize(mask_bin, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # Aplicar máscara
        masked = cv2.bitwise_and(img, img, mask=mask_bin)

        # Guardar imagen con máscara aplicada
        out_path = os.path.join(out_class, fname)
        cv2.imwrite(out_path, masked)

print(f"✅ Proceso completado. Imágenes guardadas en: {out_dir}")
