from ultralytics import YOLO
import yaml
import os
import cv2
# 1. Carga un modelo base (ej. 'yolov8n.pt' para empezar desde un modelo pre-entrenado)
#    O 'yolov8n.yaml' para empezar desde cero.
model = YOLO('yolov8n.pt')  # Cargar pesos pre-entrenados

# 2. Especifica la ruta a TU archivo .yaml
data_config_path = r"C:\Ruta\Completa\a\tu\G5 1.5.v1.yolov8 (1)\data.yaml"

def readimg(path):
  img = cv2.imread(path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (224, 224))
  return img
arr1 = []
for i in os.listdir('/content/data/train/futbol'):
  img =readimg('/content/data/train/futbol/'+i)
  arr1.append(img)

with open(data_config_path, 'r') as file:
    data = yaml.safe_load(file)

label_names = data['names']
print(label_names)

img = cv2.imread
#    YOLO leerá el data.yaml, encontrará tus carpetas train/val y comenzará.
results = model.train(
    data=data_config_path,
    epochs=100,         # Número de épocas (cuántas veces ve el dataset)
    imgsz=640,          # Tamaño de imagen
    batch=40            # Tamaño del lote (batch size)
)

# 4. (Opcional) Validar el modelo después de entrenar
results = model.val()