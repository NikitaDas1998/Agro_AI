import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("/Users/nikki/Agro_AI_/runs/classify/train/results.csv")

plt.figure()
plt.plot(results["epoch"], results["train/loss"], label="Train Loss")
plt.plot(results["epoch"], results["val/loss"], label="Val Loss", c="red")
plt.grid()
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.figure()
plt.plot(results["epoch"], results["metrics/accuracy_top1"] * 100)
plt.grid()
plt.title("Validation Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")

plt.show()