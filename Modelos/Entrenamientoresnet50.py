import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

# ===== RUTAS (ajusta si hace falta) =====
DATA_DIR     = r"C:\Users\ASUS\OneDrive\Desktop\Pulmones_Mascara"
DESKTOP_DIR  = r"C:\Users\ASUS\OneDrive\Desktop"
CKPT_PATH    = os.path.join(DESKTOP_DIR, "resnet50_best.pth")
HIST_ACC_PNG = os.path.join(DESKTOP_DIR, "resnet50_history_acc.png")
HIST_LOSS_PNG= os.path.join(DESKTOP_DIR, "resnet50_history_loss.png")
CM_PNG       = os.path.join(DESKTOP_DIR, "resnet50_confusion_matrix.png")
REPORT_TXT   = os.path.join(DESKTOP_DIR, "resnet50_metrics.txt")

# ===== CONFIG =====
batch_size  = 64          # súbelo si tienes VRAM libre
img_size    = 224
num_classes = 4
epochs      = 30
patience    = 3           # early stopping
num_workers = 4           # si da problemas, ponlo en 0

class EarlyStopping:
    def __init__(self, patience=3, mode="max", delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best = None
        self.counter = 0
        self.should_stop = False
    def step(self, metric):
        if self.best is None:
            self.best = metric
            return False
        improved = (metric > self.best + self.delta) if self.mode == "max" else (metric < self.best - self.delta)
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA disponible:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Ninguna")

    # ----- Transforms -----
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ----- Dataset & split 80/20 -----
    full_ds = datasets.ImageFolder(root=DATA_DIR, transform=transform_train)
    class_names = full_ds.classes
    train_size = int(0.8 * len(full_ds))
    val_size   = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    val_ds.dataset.transform = transform_val  # sin augmentations en val

    # DataLoaders (multiproceso en Windows con spawn)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # ----- Modelo -----
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False  # congelamos base
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    early = EarlyStopping(patience=patience, mode="max")

    # ----- Entrenamiento -----
    best_acc = 0.0
    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss, train_corrects, n_train = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(1)
            train_loss += loss.item() * inputs.size(0)
            train_corrects += (preds == labels).sum().item()
            n_train += inputs.size(0)

        epoch_train_loss = train_loss / n_train
        epoch_train_acc  = train_corrects / n_train

        # Val
        model.eval()
        val_loss, val_corrects, n_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += (preds == labels).sum().item()
                n_val += inputs.size(0)

        epoch_val_loss = val_loss / n_val
        epoch_val_acc  = val_corrects / n_val

        # Historial para gráficas
        train_acc_hist.append(epoch_train_acc)
        val_acc_hist.append(epoch_val_acc)
        train_loss_hist.append(epoch_train_loss)
        val_loss_hist.append(epoch_val_loss)

        scheduler.step(epoch_val_acc)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train: loss {epoch_train_loss:.4f}, acc {epoch_train_acc:.4f} | "
              f"Val: loss {epoch_val_loss:.4f}, acc {epoch_val_acc:.4f}")

        # Guardar mejor checkpoint
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  ✅ Mejor modelo guardado (val_acc={best_acc:.4f}) → {CKPT_PATH}")

        # Early stopping
        if early.step(epoch_val_acc):
            print(f"⛔ Early stopping activado (sin mejora en {patience} epochs).")
            break

    print(f"\nMejor val_acc lograda: {best_acc:.4f}")

    # ----- Evaluación final con el mejor ckpt -----
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            all_pred.extend(preds)
            all_true.extend(labels.numpy())

    acc = accuracy_score(all_true, all_pred)
    f1_macro    = f1_score(all_true, all_pred, average='macro')
    f1_weighted = f1_score(all_true, all_pred, average='weighted')
    report = classification_report(all_true, all_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(all_true, all_pred)

    # Guardar métricas
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("== Métricas finales (Validación) ==\n")
        f.write(f"Accuracy:      {acc:.6f}\n")
        f.write(f"F1 (macro):    {f1_macro:.6f}\n")
        f.write(f"F1 (weighted): {f1_weighted:.6f}\n\n")
        f.write("== Reporte de clasificación ==\n")
        f.write(report + "\n\n")
        f.write("== Matriz de confusión ==\n")
        f.write(np.array2string(cm))

    print(f"📝 Reporte guardado en: {REPORT_TXT}")

    # Gráficas
    plt.figure()
    plt.plot(train_acc_hist, label="Train Acc")
    plt.plot(val_acc_hist, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy")
    plt.tight_layout(); plt.savefig(HIST_ACC_PNG, dpi=200); plt.close()

    plt.figure()
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(val_loss_hist, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss")
    plt.tight_layout(); plt.savefig(HIST_LOSS_PNG, dpi=200); plt.close()

    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Matriz de Confusión'); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha='right')
    plt.yticks(ticks, class_names)
    plt.xlabel('Predicción'); plt.ylabel('Verdadero')
    plt.tight_layout(); plt.savefig(CM_PNG, dpi=200); plt.close()

    print(f"🖼️ Gráficas guardadas: {HIST_ACC_PNG}, {HIST_LOSS_PNG}, {CM_PNG}")
    print(f"✅ Mejor modelo (.pth) en: {CKPT_PATH}")

if __name__ == "__main__":
    # Arranque correcto de multiproceso en Windows
    mp.set_start_method("spawn", force=True)
    main()
