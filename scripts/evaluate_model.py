from ultralytics import YOLO

model = YOLO("/Users/nikki/Agro_AI_/runs/classify/train/weights/best.pt")
metrics = model.val()  
print("Top-1 Accuracy:", metrics.top1)
print("Top-5 Accuracy:", metrics.top5)