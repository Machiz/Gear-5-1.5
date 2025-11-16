import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import cv2

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "C:/Users/Sebastian/Desktop/PruebaModelAlexNet/dataset_combinado"
model_path = "C:/Users/Sebastian/Desktop/PruebaModelAlexNet/alexnet_gear5.pth"

# =========================================================
# FUNCIÓN: BORRAR MODELO GUARDADO (si quieres reiniciar)
# =========================================================
def borrar_modelo(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"🗑️ Modelo eliminado: {path}")
    else:
        print("⚠️ No se encontró el modelo para borrar.")

# Ejemplo: borrar_modelo(model_path)
# =========================================================


# =========================================================
# TRANSFORMACIONES Y DATASET
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
n_samples = len(dataset)
classes = dataset.classes
print(f"Clases: {classes} | Total imágenes: {n_samples}")

# =========================================================
# K-FOLD VALIDACIÓN (5 folds)
# =========================================================
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

accuracies, precisions, recalls, f1s = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"\n📂 Fold {fold + 1}/{k}")

    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)
    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=8, shuffle=False)

    # Modelo base (AlexNet)
    model = models.alexnet(weights="IMAGENET1K_V1")
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Entrenamiento rápido por fold
    model.train()
    for epoch in range(5):  # menos épocas por fold para velocidad
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  Época {epoch+1}/5 | Pérdida: {running_loss/len(train_loader):.4f}")

    # Evaluación
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    accuracies.append(report["accuracy"])
    precisions.append(np.mean([report[c]["precision"] for c in classes]))
    recalls.append(np.mean([report[c]["recall"] for c in classes]))
    f1s.append(np.mean([report[c]["f1-score"] for c in classes]))

    print(classification_report(all_labels, all_preds, target_names=classes))

    # MATRIZ DE CONFUSIÓN por fold
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Matriz de Confusión - Fold {fold+1}")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

# =========================================================
# RESULTADOS PROMEDIO
# =========================================================
print("\n📊 RESULTADOS PROMEDIO DE LOS 5 FOLDS:")
print(f"Accuracy promedio: {np.mean(accuracies):.3f}")
print(f"Precision promedio: {np.mean(precisions):.3f}")
print(f"Recall promedio: {np.mean(recalls):.3f}")
print(f"F1-score promedio: {np.mean(f1s):.3f}")

plt.figure(figsize=(7,5))
plt.plot(range(1, k+1), accuracies, marker='o', label="Accuracy")
plt.plot(range(1, k+1), f1s, marker='s', label="F1-score")
plt.title("Resultados por Fold - AlexNet (KFold 5)")
plt.xlabel("Fold")
plt.ylabel("Métrica")
plt.legend()
plt.grid(True)
plt.show()


# =========================================================
# GRAD-CAM VISUALIZACIÓN
# =========================================================
def generate_gradcam(model, img_path, target_layer="features.12"):
    model.eval()
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform_norm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform_norm(transforms.ToPILImage()(img)).unsqueeze(0).to(device)

    def forward_hook(module, input, output):
        nonlocal conv_output
        conv_output = output

    conv_output = None
    hook = dict(model.named_modules())[target_layer].register_forward_hook(forward_hook)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    score = output[0, pred_class]
    model.zero_grad()
    score.backward()

    grads = model.features[12].weight.grad
    weights = torch.mean(grads, dim=(1, 2, 3))
    cam = torch.zeros(conv_output.shape[2:], dtype=torch.float32)

    for i, w in enumerate(weights):
        cam += w * conv_output[0, i, :, :].cpu().detach()

    cam = torch.clamp(cam, min=0)
    cam /= torch.max(cam)
    cam = cv2.resize(cam.numpy(), (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM - Predicción: {classes[pred_class]}")
    plt.axis("off")
    plt.show()

# Ejemplo: (ajusta la ruta a una imagen tuya)
# generate_gradcam(model, "C:/Users/Sebastian/Desktop/PruebaModelAlexNet/dataset_combinado/dañadasN/cartaDañada-001.png")
