# -*- coding: utf-8 -*-
# =====================================================
#  MODELO ALEXNET - PROYECTO GEAR 5 (100 imágenes)
# =====================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# ============================
# 1️⃣ CONFIGURACIÓN INICIAL
# ============================

# Unir ambas carpetas (normal y dañada) en un solo dataset temporal
data_dir = r"C:\Users\Sebastian\Desktop\PruebaModelAlexNet\dataset_combinado"

# Crear el dataset combinado si no existe
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    os.makedirs(os.path.join(data_dir, "dañadasN"))
    os.makedirs(os.path.join(data_dir, "normalN"))

    # Copiar imágenes desde las rutas originales
    import shutil
    src_danadas = r"C:\Users\Sebastian\Desktop\PruebaModelAlexNet\dañadasN-20251027T011456Z-1-001\dañadasN"
    src_normales = r"C:\Users\Sebastian\Desktop\PruebaModelAlexNet\normalN-20251027T011533Z-1-001\normalN"

    # Copiar solo las primeras 50 de cada clase
    for i, img in enumerate(os.listdir(src_danadas)[:50]):
        shutil.copy(os.path.join(src_danadas, img), os.path.join(data_dir, "dañadasN", img))
    for i, img in enumerate(os.listdir(src_normales)[:50]):
        shutil.copy(os.path.join(src_normales, img), os.path.join(data_dir, "normalN", img))

print("📁 Dataset combinado preparado en:", data_dir)

# ============================
# 2️⃣ PARÁMETROS
# ============================
batch_size = 16
num_classes = 2
num_epochs = 10
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Usando dispositivo: {device}")

# ============================
# 3️⃣ TRANSFORMACIONES
# ============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================
# 4️⃣ CARGA DEL DATASET
# ============================
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
print("Clases detectadas:", dataset.classes)

# Verificar conteo
labels = [label for _, label in dataset.samples]
print("Conteo total por clase:", Counter(labels))

# Dividir en 80% train y 20% test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"📂 Dataset total: {len(dataset)} imágenes (Train: {train_size}, Test: {test_size})")

# ============================
# 5️⃣ MODELO ALEXNET
# ============================
model = models.alexnet(weights='IMAGENET1K_V1')

# Congelar capas base
for param in model.parameters():
    param.requires_grad = False

# Reemplazar la capa final para 2 clases
model.classifier[6] = nn.Linear(4096, num_classes)
model = model.to(device)

# ============================
# 6️⃣ ENTRENAMIENTO
# ============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=learning_rate)

train_losses = []

print("\n🚀 Iniciando entrenamiento...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"📈 Época [{epoch+1}/{num_epochs}] - Pérdida: {avg_loss:.4f}")

print("✅ Entrenamiento completado.")

# ============================
# 7️⃣ EVALUACIÓN
# ============================
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n📊 MÉTRICAS DEL MODELO ALEXNET:")
target_names = dataset.classes
print(classification_report(y_true, y_pred, target_names=target_names))

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - AlexNet (Gear 5)")
plt.show()

# Gráfico de pérdida
plt.figure(figsize=(6, 4))
plt.plot(train_losses, marker='o', label="Pérdida de entrenamiento")
plt.title("Evolución de la pérdida (AlexNet)")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.show()
