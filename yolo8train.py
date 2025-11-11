from ultralytics import YOLO

def main():
    model = YOLO("yolo8n.yaml")

    data_yaml_file = r"D:\Gear-5-1.5\G5 1.5.v1i.yolov8 (1)\data.yaml"
    project = r"D:\Gear-5-1.5\G5 1.5.v1i.yolov8 (1)"
    experiment = "My-card-model"

    batch_size = 32

    results = model.train(data = data_yaml_file,
                          epochs = 100,
                          project = project,
                          name = experiment,
                          batch = batch_size,
                          device = 0,
                          patience = 5,
                          imgsz = 640,
                          verbose = True,
                          val = True)
    
if __name__ == "__main__":
    main()
    