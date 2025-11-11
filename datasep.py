import os
import shutil
import random
import glob

# --- Configuración (¡DEBES EDITAR ESTO!) ---

# 1. Ruta a tu carpeta 'train' original (la que contiene 'images' y 'labels')
#    Usa 'r' antes de la ruta para evitar problemas con '\' en Windows.
#    Ejemplo: r"C:\Usuarios\TuNombre\Descargas\G5 1.5.v1.yolov8 (1)\train"
SOURCE_DIR = r"C:\Users\marce\Gear-5-1.5\G5 1.5.v1i.yolov8 (1)\train"

# 2. Ruta donde se creará la nueva estructura 'dataset_split'
#    Por defecto, la crea un nivel "arriba" de tu SOURCE_DIR.
#    Ej: ...\Descargas\G5 1.5.v1.yolov8 (1)\dataset_split
OUTPUT_DIR = os.path.join(os.path.dirname(SOURCE_DIR), "dataset_split")

# 3. Proporciones (asegúrate de que sumen 1.0)
TRAIN_RATIO = 0.8  # 80% para entrenamiento
VAL_RATIO = 0.1    # 10% para validación
TEST_RATIO = 0.1   # 10% para prueba

# 4. Extensiones de archivo (¡IMPORTANTE!)
#    Asegúrate de que coincida con tus archivos (ej: ".jpg", ".png")
IMAGE_EXTENSION = ".jpg" 
LABEL_EXTENSION = ".txt" # Extensión de las etiquetas (generalmente .txt para YOLO)

# --- Fin de la Configuración ---


def create_split_dirs(base_dir):
    """
    Crea la estructura de carpetas train/val/test con subcarpetas images/labels.
    Si la carpeta 'base_dir' ya existe, la elimina y la recrea.
    """
    if os.path.exists(base_dir):
        print(f"Advertencia: La carpeta '{base_dir}' ya existe. Se eliminará y recreará.")
        shutil.rmtree(base_dir)

    print(f"Creando estructura de carpetas en: {base_dir}")
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)

def split_dataset():
    """
    Encuentra todos los archivos, los divide y los copia a las carpetas de destino.
    """
    
    source_images_dir = os.path.join(SOURCE_DIR, 'images')
    source_labels_dir = os.path.join(SOURCE_DIR, 'labels')

    # 1. Validar rutas de origen
    if not os.path.isdir(source_images_dir) or not os.path.isdir(source_labels_dir):
        print(f"Error: No se encontraron las carpetas 'images' o 'labels' en '{SOURCE_DIR}'")
        print("Por favor, edita la variable 'SOURCE_DIR' en el script.")
        return
        
    # 2. Crear carpetas de destino
    create_split_dirs(OUTPUT_DIR)

    # 3. Obtener la lista de archivos de imagen
    # Usamos glob para encontrar todos los archivos con la extensión correcta
    image_files = glob.glob(os.path.join(source_images_dir, f"*{IMAGE_EXTENSION}"))
    
    if not image_files:
        print(f"Error: No se encontraron imágenes con extensión '{IMAGE_EXTENSION}' en '{source_images_dir}'")
        return
        
    print(f"Total de imágenes encontradas: {len(image_files)}")
    
    # 4. Mezclar los archivos
    random.seed(42) # Usamos una semilla para que la división sea reproducible
    random.shuffle(image_files)

    # 5. Calcular los puntos de división
    total_count = len(image_files)
    train_count = int(total_count * TRAIN_RATIO)
    val_count = int(total_count * VAL_RATIO)
    
    # El resto va a 'test'
    train_files = image_files[:train_count]
    val_files = image_files[train_count : train_count + val_count]
    test_files = image_files[train_count + val_count :]

    # 6. Función interna para copiar los archivos
    def copy_files_to_split(file_list, split_name):
        """Copia un par (imagen, etiqueta) a la carpeta de destino (train/val/test)."""
        print(f"Copiando {len(file_list)} archivos a '{split_name}'...")
        for img_path in file_list:
            
            # --- Encontrar el par imagen-etiqueta ---
            # Nombre base (ej: 'imagen_001')
            base_filename = os.path.basename(img_path)
            base_name = os.path.splitext(base_filename)[0]
            
            # Nombre del archivo de etiqueta (ej: 'imagen_001.txt')
            label_filename = base_name + LABEL_EXTENSION
            label_path = os.path.join(source_labels_dir, label_filename)
            
            # --- Definir rutas de destino ---
            dest_img_path = os.path.join(OUTPUT_DIR, split_name, 'images', base_filename)
            dest_label_path = os.path.join(OUTPUT_DIR, split_name, 'labels', label_filename)

            # --- Copiar ---
            # Verificar si la etiqueta correspondiente existe antes de copiar
            if os.path.exists(label_path):
                shutil.copy(img_path, dest_img_path)
                shutil.copy(label_path, dest_label_path)
            else:
                print(f"  Advertencia: No se encontró la etiqueta '{label_filename}' para la imagen '{base_filename}'. Se omitirá este par.")

    # 7. Ejecutar la copia para cada conjunto
    copy_files_to_split(train_files, 'train')
    copy_files_to_split(val_files, 'val')
    copy_files_to_split(test_files, 'test')

    print("\n--- Proceso Completado ---")
    print(f"Total de imágenes procesadas: {total_count}")
    print(f"Archivos de entrenamiento: {len(train_files)}")
    print(f"Archivos de validación:  {len(val_files)}")
    print(f"Archivos de prueba:      {len(test_files)}")
    print(f"\nDataset dividido creado exitosamente en: {OUTPUT_DIR}")

# --- Ejecutar el script ---
if __name__ == "__main__":
    
    # Pequeña comprobación de seguridad
    if SOURCE_DIR == r"C:\Ruta\Completa\A\Tu\Carpeta\train":
        print("¡Error! Debes editar la variable 'SOURCE_DIR' en el script.")
        print("Abre el archivo .py y cambia la ruta a la de tu carpeta 'train'.")
    else:
        # Validar que las proporciones sumen 1 (con un pequeño margen de error)
        if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-9:
            print("Error: Las proporciones (RATIO) no suman 1.0. Revisa la configuración.")
        else:
            split_dataset()