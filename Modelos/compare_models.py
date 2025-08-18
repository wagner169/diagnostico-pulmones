import io
from pathlib import Path
import gdown
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ---- Config ----
FILES = {
    "resnet50": "1yhdFMGyaw-gW4yU5pPl6CFJRZVvS44np",
    "densenet121": "1KpLhlrcAgSaIUIIABNpu6LzFGGG-a2TS",
    "efficientnet_b0": "1FuliZMAQdaiOWYlsahiYJqzc8fy_2oUQ",
}
CLASS_NAMES = ["Normal", "Opacidad Pulmonar", "Neumonía", "COVID-19"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ---- Descarga pesos ----
for name, fid in FILES.items():
    out = MODELS_DIR / f"{name}.pth"
    if not out.exists():
        url = f"https://drive.google.com/uc?id={fid}"
        print(f"⬇ Descargando {name} ...")
        gdown.download(url, str(out), quiet=False)

# ---- Preprocesado ----
IMG_TFMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def _strip_module(state_dict):
    if any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def load_models():
    resnet = models.resnet50(weights=None)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, len(CLASS_NAMES))

    densenet = models.densenet121(weights=None)
    densenet.classifier = torch.nn.Linear(densenet.classifier.in_features, len(CLASS_NAMES))

    efficient = models.efficientnet_b0(weights=None)
    efficient.classifier[1] = torch.nn.Linear(efficient.classifier[1].in_features, len(CLASS_NAMES))

    models_dict = {
        "resnet50": resnet,
        "densenet121": densenet,
        "efficientnet_b0": efficient
    }

    for name, model in models_dict.items():
        state = torch.load(MODELS_DIR / f"{name}.pth", map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(_strip_module(state), strict=False)
        model.eval().to(DEVICE)

    return models_dict

def predict_best(image_path: str):
    img = Image.open(image_path).convert("RGB")
    x = IMG_TFMS(img).unsqueeze(0).to(DEVICE)

    models_dict = load_models()
    results = {}
    for name, model in models_dict.items():
        with torch.inference_mode():
            p = F.softmax(model(x), dim=1).squeeze(0).cpu().tolist()
            results[name] = {cls: float(p[i]) for i, cls in enumerate(CLASS_NAMES)}

    # elegir el modelo más seguro en su predicción
    best_model = max(results, key=lambda m: max(results[m].values()))
    best_class = max(results[best_model], key=results[best_model].get)
    return best_model, best_class, results[best_model]

# ---- Ejemplo ----
if __name__ == "__main__":
    modelo, clase, probs = predict_best("ejemplo.jpg")
    print(f"🏆 Mejor modelo: {modelo}")
    print(f"✅ Predicción: {clase}")
    print("Probs:", probs)
