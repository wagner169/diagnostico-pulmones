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

# ===== RUTAS =====
DATA_DIR     = os.path.join('data', 'Pulmones_Mascara')
OUTPUTS_DIR  = 'outputs'
os.makedirs(OUTPUTS_DIR, exist_ok=True)
CKPT_PATH    = os.path.join(OUTPUTS_DIR, 'resnet50_best.pth')
HIST_ACC_PNG = os.path.join(OUTPUTS_DIR, 'resnet50_history_acc.png')
HIST_LOSS_PNG= os.path.join(OUTPUTS_DIR, 'resnet50_history_loss.png')
CM_PNG       = os.path.join(OUTPUTS_DIR, 'resnet50_confusion_matrix.png')
REPORT_TXT   = os.path.join(OUTPUTS_DIR, 'resnet50_metrics.txt')

# ===== CONFIG =====
batch_size  = 64
img_size    = 224
epochs      = 30
patience    = 3
num_workers = 4
seed        = 42

class EarlyStopping:
    def __init__(self, patience=3, mode='max', delta=0.0):
        self.patience, self.mode, self.delta = patience, mode, delta
        self.best, self.counter, self.should_stop = None, 0, False
    def step(self, metric):
        if self.best is None: self.best = metric; return False
        improved = (metric > self.best + self.delta) if self.mode=='max' else (metric < self.best - self.delta)
        if improved: self.best = metric; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.should_stop = True
        return self.should_stop

def main():
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CUDA disponible:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))

    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    full_ds = datasets.ImageFolder(DATA_DIR, transform=tf_train)
    class_names = full_ds.classes
    n_train = int(0.8 * len(full_ds))
    n_val   = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    val_ds.dataset.transform = tf_val

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for p in model.parameters(): p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, len(class_names))
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    early = EarlyStopping(patience=patience, mode='max')

    best_acc = 0.0
    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []

    for epoch in range(1, epochs + 1):
        model.train(); tr_loss=tr_corr=tr_n=0
        for x,y in train_ld:
            x,y = x.to(device,non_blocking=True), y.to(device,non_blocking=True)
            optimizer.zero_grad(); out = model(x)
            loss = criterion(out,y)
            loss.backward(); optimizer.step()
            tr_loss += loss.item()*x.size(0)
            tr_corr += (out.argmax(1)==y).sum().item(); tr_n += x.size(0)
        tr_loss /= tr_n; tr_acc = tr_corr/tr_n

        model.eval(); va_loss=va_corr=va_n=0
        with torch.no_grad():
            for x,y in val_ld:
                x,y = x.to(device,non_blocking=True), y.to(device,non_blocking=True)
                out = model(x); loss = criterion(out,y)
                va_loss += loss.item()*x.size(0)
                va_corr += (out.argmax(1)==y).sum().item(); va_n += x.size(0)
        va_loss /= va_n; va_acc = va_corr/va_n

        train_acc_hist.append(tr_acc); val_acc_hist.append(va_acc)
        train_loss_hist.append(tr_loss); val_loss_hist.append(va_loss)
        scheduler.step(va_acc)

        print(f"Epoch {epoch:02d}/{epochs} | Train: loss {tr_loss:.4f}, acc {tr_acc:.4f} | Val: loss {va_loss:.4f}, acc {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  ✅ Mejor modelo guardado (val_acc={best_acc:.4f}) → {CKPT_PATH}")

        if early.step(va_acc):
            print(f"⛔ Early stopping (paciencia={patience})."); break

    print(f"\nMejor val_acc: {best_acc:.4f}")

    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x,y in val_ld:
            x = x.to(device,non_blocking=True)
            logits = model(x)
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro')
    f1w = f1_score(y_true, y_pred, average='weighted')
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm  = confusion_matrix(y_true, y_pred)

    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write('== Métricas finales (Validación) ==\n')
        f.write(f'Accuracy:      {acc:.6f}\n')
        f.write(f'F1 (macro):    {f1m:.6f}\n')
        f.write(f'F1 (weighted): {f1w:.6f}\n\n')
        f.write('== Reporte de clasificación ==\n')
        f.write(rep + '\n\n')
        f.write('== Matriz de confusión ==\n')
        f.write(np.array2string(cm))

    plt.figure(); plt.plot(train_acc_hist, label='Train Acc'); plt.plot(val_acc_hist, label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy')
    plt.tight_layout(); plt.savefig(HIST_ACC_PNG, dpi=200); plt.close()

    plt.figure(); plt.plot(train_loss_hist, label='Train Loss'); plt.plot(val_loss_hist, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')
    plt.tight_layout(); plt.savefig(HIST_LOSS_PNG, dpi=200); plt.close()

    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest'); plt.title('Matriz de Confusión - ResNet50'); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha='right'); plt.yticks(ticks, class_names)
    plt.xlabel('Predicción'); plt.ylabel('Verdadero')
    plt.tight_layout(); plt.savefig(CM_PNG, dpi=200); plt.close()

    print(f"📝 Reporte → {REPORT_TXT}")
    print(f"🖼️ Guardados → {HIST_ACC_PNG}, {HIST_LOSS_PNG}, {CM_PNG}")
    print(f"✅ Checkpoint → {CKPT_PATH}")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
