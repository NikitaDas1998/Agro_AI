import sys
from ultralytics import YOLO
import numpy as np

def predict_disease(image_path):
    model = YOLO("runs/classify/train/weights/best.pt")
    results = model(image_path)
    names = results[0].names
    probs = results[0].probs.data.tolist()
    predicted = names[np.argmax(probs)]
    print("✅ Predicted Disease:", predicted)
    return predicted

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Please provide image path: python predict_image.py path/to/image.jpg")
    else:
        detect_disease(sys.argv[1])