from ultralytics import YOLO

model = YOLO("models/yolov12-cls.yaml")  

model.train(
    data="/Users/nikki/Agro_AI_/Dataset",    
    epochs=100,
    imgsz=64,
    task="classify"
)