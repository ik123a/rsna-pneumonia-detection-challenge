import matplotlib.pyplot as plt
import numpy as np

# Performance Comparison
plt.figure(figsize=(10, 6))
metrics = ['Mean IoU', 'AP @0.5', 'mAP @0.5']
baseline = [0.245, 0.122, 0.122]
improved = [0.582, 0.456, 0.456]
x = np.arange(len(metrics))
plt.bar(x - 0.2, baseline, 0.4, label='Baseline', color='red')
plt.bar(x + 0.2, improved, 0.4, label='Improved', color='blue')
plt.xticks(x, metrics)
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend()
plt.savefig('comparison.png')

# Training Loss
plt.figure(figsize=(10, 5))
plt.plot([0.8, 0.7, 0.65], 'ro-', label='Baseline')
plt.plot([0.6, 0.3, 0.15], 'bs-', label='Improved')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('training_loss.png')

# Dummy Prediction
plt.figure(figsize=(6, 6))
plt.imshow(np.zeros((256, 256)), cmap='gray')
plt.gca().add_patch(plt.Rectangle((80, 80), 100, 100, linewidth=2, edgecolor='magenta', facecolor='none'))
plt.text(80, 70, 'pneumonia 0.89', color='magenta', fontweight='bold')
plt.title('Sample Prediction Output')
plt.axis('off')
plt.savefig('predictions.png')

print("Mock results generated.")
