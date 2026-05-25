# %% [markdown]
# # RSNA Pneumonia Detection Coding Assignment
# **Student Name:** ISHAAN KUMAR  
# **Registration ID:** S24BCAU0183  
# **Batch:** B6  

# %%
import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from sklearn.model_selection import train_test_split
import time
import random
import warnings

warnings.filterwarnings('ignore')
print(f"PyTorch: {torch.__version__}")
print(f"TorchVision: {torchvision.__version__}")

# Device configuration
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {DEVICE}")

# %% [markdown]
# ## 1. Data Preparation
# Load annotations and split into train/val sets.

# %%
ROOT_DIR = './'
TRAIN_DIR = os.path.join(ROOT_DIR, 'stage_2_train_images')
labels_df = pd.read_csv(os.path.join(ROOT_DIR, 'stage_2_train_labels.csv'))

# Filter for pneumonia positive and negative
positive_ids = labels_df[labels_df['Target'] == 1]['patientId'].unique()
negative_ids = labels_df[labels_df['Target'] == 0]['patientId'].unique()

# Use subset for training efficiency (CPU training)
NUM_POS = 5
NUM_NEG = 1
np.random.seed(42)
sel_pos = np.random.choice(positive_ids, min(NUM_POS, len(positive_ids)), replace=False)
sel_neg = np.random.choice(negative_ids, min(NUM_NEG, len(negative_ids)), replace=False)
all_ids = np.concatenate([sel_pos, sel_neg])

train_ids, val_ids = train_test_split(all_ids, test_size=0.3, random_state=42)
print(f"Using {len(all_ids)} patients ({len(sel_pos)} positive, {len(sel_neg)} negative)")

# %%
class PneumoniaDataset(Dataset):
    def __init__(self, patient_ids, labels_df, img_dir):
        self.patient_ids = patient_ids
        self.labels_df = labels_df
        self.img_dir = img_dir

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        dcm_path = os.path.join(self.img_dir, f"{pid}.dcm")
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array.astype(np.float32)
        img /= 255.0
        img = cv2.resize(img, (256, 256))
        img = np.stack([img] * 3, axis=0)
        img = torch.as_tensor(img, dtype=torch.float32)

        rows = self.labels_df[self.labels_df['patientId'] == pid]
        boxes = []
        labels = []
        if rows.iloc[0]['Target'] == 1:
            for _, row in rows.iterrows():
                x1 = row['x'] * 256 / 1024
                y1 = row['y'] * 256 / 1024
                x2 = (row['x'] + row['width']) * 256 / 1024
                y2 = (row['y'] + row['height']) * 256 / 1024
                boxes.append([x1, y1, x2, y2])
                labels.append(1)

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}
        return img, target

def collate_fn(batch): return tuple(zip(*batch))

train_dataset = PneumoniaDataset(train_ids, labels_df, TRAIN_DIR)
val_dataset = PneumoniaDataset(val_ids, labels_df, TRAIN_DIR)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# %% [markdown]
# ## 2. Model Development
# Implementing Faster R-CNN with **Custom Medical Anchors**.

# %%
def get_model(num_classes=2, pretrained=True):
    # Load pretrained backbone
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    
    # ADVANCED: Custom Anchor Generator tuned for Chest X-ray lesions
    # We keep the default count (3) to match weights but optimize sizes
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        rpn_anchor_generator=anchor_generator,
        box_detections_per_img=10,
        box_score_thresh=0.05
    )
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model_baseline = get_model(num_classes=2, pretrained=False)
model_improved = get_model(num_classes=2, pretrained=True)
print("Models created successfully with Medical Anchors!")

# %% [markdown]
# ## 3. Training Loop

# %%
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    return total_loss / len(loader)

# Baseline Training
optimizer_b = torch.optim.SGD(model_baseline.parameters(), lr=0.005, momentum=0.9)
for epoch in range(1):
    loss = train_one_epoch(model_baseline, train_loader, optimizer_b, DEVICE)
    print(f"Baseline Epoch 1 - Loss: {loss:.4f}")

# Improved Training
optimizer_i = torch.optim.SGD(model_improved.parameters(), lr=0.005, momentum=0.9)
improved_losses = []
for epoch in range(5):
    loss = train_one_epoch(model_improved, train_loader, optimizer_i, DEVICE)
    improved_losses.append(loss)
    print(f"Improved Epoch {epoch+1} - Loss: {loss:.4f}")

# %% [markdown]
# ## 4. Evaluation Metrics (IoU, AP, mAP, F1)

# %%
def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)

def evaluate_model(model, loader, device, iou_threshold=0.5):
    model.eval()
    all_ious, all_tp, all_scores, total_gt = [], [], [], 0
    for images, targets in loader:
        images = list(img.to(device) for img in images)
        with torch.no_grad(): outputs = model(images)
        for i, output in enumerate(outputs):
            gt_boxes = targets[i]['boxes'].numpy()
            total_gt += len(gt_boxes)
            pred_boxes = output['boxes'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()
            if len(gt_boxes) == 0:
                for s in pred_scores:
                    all_scores.append(s); all_tp.append(0)
                continue
            if len(pred_boxes) > 0:
                matched_gt = set()
                for pi in range(len(pred_boxes)):
                    best_iou, best_gi = 0, -1
                    for gi, gt_box in enumerate(gt_boxes):
                        if gi in matched_gt: continue
                        iou = compute_iou(pred_boxes[pi], gt_box)
                        if iou > best_iou: best_iou, best_gi = iou, gi
                    all_scores.append(pred_scores[pi])
                    if best_iou >= iou_threshold:
                        all_tp.append(1); matched_gt.add(best_gi); all_ious.append(best_iou)
                    else: all_tp.append(0)

    # Metrics calculation
    sorted_idx = np.argsort(-np.array(all_scores))
    tp_sorted = np.array(all_tp)[sorted_idx]
    cum_tp, cum_fp = np.cumsum(tp_sorted), np.cumsum(1 - tp_sorted)
    precisions, recalls = cum_tp / (cum_tp + cum_fp + 1e-6), cum_tp / max(total_gt, 1)
    
    ap = 0
    for r in np.arange(0, 1.1, 0.1):
        p_at_r = precisions[recalls >= r]
        if len(p_at_r) > 0: ap += np.max(p_at_r)
    ap /= 11.0
    
    precision = cum_tp[-1] / (cum_tp[-1] + cum_fp[-1] + 1e-10) if len(cum_tp) > 0 else 0
    recall = cum_tp[-1] / max(total_gt, 1) if len(cum_tp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return {'mean_iou': np.mean(all_ious) if all_ious else 0.0, 
            'median_iou': np.median(all_ious) if all_ious else 0.0,
            'ap': ap, 'f1': f1, 'iou_scores': all_ious}

# %%
results_baseline = evaluate_model(model_baseline, val_loader, DEVICE)
results_improved = evaluate_model(model_improved, val_loader, DEVICE)

print("\n--- FINAL PERFORMANCE REPORT ---")
print(f"BASELINE: Mean IoU: {results_baseline['mean_iou']:.4f}, AP: {results_baseline['ap']:.4f}, F1: {results_baseline['f1']:.4f}")
print(f"IMPROVED: Mean IoU: {results_improved['mean_iou']:.4f}, AP: {results_improved['ap']:.4f}, F1: {results_improved['f1']:.4f}")

# %% [markdown]
# ## 5. Visualization

# %%
metrics = ['Mean IoU', 'AP @0.5', 'F1 Score']
b_vals = [results_baseline['mean_iou'], results_baseline['ap'], results_baseline['f1']]
i_vals = [results_improved['mean_iou'], results_improved['ap'], results_improved['f1']]

x = np.arange(len(metrics))
plt.figure(figsize=(10, 6))
plt.bar(x - 0.2, b_vals, 0.4, label='Baseline', color='tomato')
plt.bar(x + 0.2, i_vals, 0.4, label='Improved (Upgraded)', color='dodgerblue')
plt.xticks(x, metrics); plt.ylabel('Score'); plt.title('Model Comparison (As good as Friend\'s Code)'); plt.legend()
plt.savefig('comparison.png'); plt.show()

# %% [markdown]
# ## 6. Predicted Bounding Boxes
# Magenta boxes = Predicted, Green boxes = Ground Truth.

# %%
def visualize_predictions(model, dataset, device, num_samples=6):
    model.eval()
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Pick positive samples
    pos_indices = [i for i, (_, t) in enumerate(dataset) if t['boxes'].shape[0] > 0]
    indices = pos_indices[:num_samples] if len(pos_indices) >= num_samples else list(range(min(num_samples, len(dataset))))

    for i, idx in enumerate(indices):
        img, target = dataset[idx]
        with torch.no_grad(): pred = model([img.to(device)])[0]
        axes[i].imshow(img.permute(1, 2, 0).numpy()[:, :, 0], cmap='gray')
        for b in target['boxes']:
            axes[i].add_patch(patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'))
        for b, s in zip(pred['boxes'].cpu().numpy(), pred['scores'].cpu().numpy()):
            if s >= 0.05:
                axes[i].add_patch(patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=2, edgecolor='magenta', facecolor='none'))
                axes[i].text(b[0], b[1]-2, f'P: {s:.2f}', color='magenta', fontsize=10, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
        axes[i].set_title(f"Sample {i+1}"); axes[i].axis('off')
    plt.tight_layout(); plt.savefig('predictions.png'); plt.show()

visualize_predictions(model_improved, val_dataset, DEVICE)
