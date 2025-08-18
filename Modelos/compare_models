import os
import re
import pandas as pd

OUTPUTS_DIR = 'outputs'
files = {
    'ResNet-50':       os.path.join(OUTPUTS_DIR, 'resnet50_metrics.txt'),
    'DenseNet-121':    os.path.join(OUTPUTS_DIR, 'densenet121_metrics.txt'),
    'EfficientNet-B0': os.path.join(OUTPUTS_DIR, 'efficientnet_b0_metrics.txt'),
}

rows = []
for name, path in files.items():
    if not os.path.exists(path):
        print(f"⚠️  No existe: {path}")
        rows.append({'Modelo': name, 'Accuracy': None, 'F1 Macro': None, 'F1 Weighted': None})
        continue
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    acc = re.search(r"Accuracy:\s*([\d.]+)", text)
    f1m = re.search(r"F1\s*\(macro\):\s*([\d.]+)", text)
    f1w = re.search(r"F1\s*\(weighted\):\s*([\d.]+)", text)
    rows.append({
        'Modelo': name,
        'Accuracy': float(acc.group(1)) if acc else None,
        'F1 Macro': float(f1m.group(1)) if f1m else None,
        'F1 Weighted': float(f1w.group(1)) if f1w else None,
    })

out = pd.DataFrame(rows).sort_values(by=['Accuracy','F1 Macro','F1 Weighted'], ascending=False)
print(out)

os.makedirs(OUTPUTS_DIR, exist_ok=True)
out.to_excel(os.path.join(OUTPUTS_DIR, 'comparacion_modelos.xlsx'), index=False)
out.to_csv(os.path.join(OUTPUTS_DIR, 'comparacion_modelos.csv'), index=False, encoding='utf-8', float_format='%.6f')
print('✅ Guardado compare: outputs/comparacion_modelos.xlsx y .csv')
