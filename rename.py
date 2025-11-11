import os
import glob

# --- CONFIGURACIÓN (¡EDITA ESTO!) ---

# 1. Ruta a la carpeta que contiene 'images' y 'labels'
#    (ej. la carpeta 'train', 'val', o 'test')
#
#    Ejemplo: r"C:\Usuarios\TuNombre\Descargas\G5 1.5.v1.yolov8 (1)\train"
#    OJO: Si tienes train, val y test, debes correr el script 3 veces,
#    cambiando esta ruta cada vez.
BASE_DIR = r"C:\Ruta\Completa\a\tu\carpeta\set" 

# 2. Prefijo para el nuevo nombre
NEW_BASENAME = "imagen"

# 3. Número por el que empezar a contar
START_INDEX = 1

# 4. Sufijo temporal (no lo cambies a menos que sepas lo que haces)
TEMP_SUFFIX = "_temp_rename_12345"
# --- FIN DE LA CONFIGURACIÓN ---


def rename_files_robustly():
    """
    Renombra archivos en dos fases para evitar colisiones de nombres.
    Fase 1: nombre_original -> nombre_original_temp
    Fase 2: nombre_original_temp -> nombre_nuevo (ej. imagen1)
    """
    images_dir = os.path.join(BASE_DIR, 'images')
    labels_dir = os.path.join(BASE_DIR, 'labels')

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"Error: No se encuentran 'images' o 'labels' en '{BASE_DIR}'")
        print("Asegúrate de que la ruta 'BASE_DIR' apunte a una carpeta")
        print("que contenga directamente las subcarpetas 'images' y 'labels'.")
        return

    # --- FASE 1: Renombrar a temporal ---
    print(f"Buscando imágenes en: {images_dir}")
    image_files = []
    # Buscar extensiones comunes
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))

    # Filtrar archivos que ya son temporales (si el script falló a la mitad)
    image_files = [f for f in image_files if TEMP_SUFFIX not in f]
    
    if not image_files:
        print("No se encontraron imágenes para la Fase 1.")
    else:
        print(f"FASE 1: Renombrando {len(image_files)} pares de archivos a nombres temporales...")
        image_files.sort() 

        for old_img_path in image_files:
            img_extension = os.path.splitext(old_img_path)[1]
            old_base_name = os.path.splitext(os.path.basename(old_img_path))[0]
            old_label_path = os.path.join(labels_dir, old_base_name + ".txt")

            if not os.path.exists(old_label_path):
                print(f"  -> ADVERTENCIA: Se omite '{old_img_path}'. No se encontró la etiqueta '{old_label_path}'.")
                continue

            # Nombres temporales
            temp_img_name = f"{old_base_name}{TEMP_SUFFIX}{img_extension}"
            temp_label_name = f"{old_base_name}{TEMP_SUFFIX}.txt"
            temp_img_path = os.path.join(images_dir, temp_img_name)
            temp_label_path = os.path.join(labels_dir, temp_label_name)

            try:
                os.rename(old_img_path, temp_img_path)
                os.rename(old_label_path, temp_label_path)
                print(f"  {old_base_name} -> {old_base_name}{TEMP_SUFFIX}")
            except Exception as e:
                print(f"Error en Fase 1 renombrando {old_base_name}: {e}")
                return # Detener si hay un error

    print("\nFASE 2: Renombrando archivos temporales a nombres finales...")
    
    # Buscar TODOS los archivos temporales (imágenes)
    temp_image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp']:
        temp_image_files.extend(glob.glob(os.path.join(images_dir, f"*{TEMP_SUFFIX}{ext}")))
    
    temp_image_files.sort()
    
    if not temp_image_files:
        print("No se encontraron archivos temporales para la Fase 2.")
        return

    counter = START_INDEX
    for temp_img_path in temp_image_files:
        img_extension = os.path.splitext(temp_img_path)[1]
        temp_base_name = os.path.splitext(os.path.basename(temp_img_path))[0]
        
        # Ruta de la etiqueta temporal
        temp_label_path = os.path.join(labels_dir, temp_base_name + ".txt")

        if not os.path.exists(temp_label_path):
            print(f"  -> ADVERTENCIA: Se omite '{temp_img_path}'. No se encontró la etiqueta temporal.")
            continue

        # Nombres finales
        new_img_name = f"{NEW_BASENAME}{counter}{img_extension}"
        new_label_name = f"{NEW_BASENAME}{counter}.txt"
        new_img_path = os.path.join(images_dir, new_img_name)
        new_label_path = os.path.join(labels_dir, new_label_name)
        
        try:
            os.rename(temp_img_path, new_img_path)
            os.rename(temp_label_path, new_label_path)
            print(f"  {temp_base_name} -> {NEW_BASENAME}{counter}")
            counter += 1
        except Exception as e:
            print(f"Error en Fase 2 renombrando {temp_base_name}: {e}")

    print(f"\n¡Proceso completado! {counter - START_INDEX} pares de archivos renombrados.")

# --- Ejecutar el script ---
if __name__ == "__main__":
    if BASE_DIR == r"C:\Ruta\Completa\a\tu\carpeta\set":
        print("¡ERROR! Abre el script .py y edita la variable 'BASE_DIR'.")
        print("Debe apuntar a tu carpeta 'train', 'val' o 'test'.")
    else:
        rename_files_robustly()