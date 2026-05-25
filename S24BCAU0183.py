# %% [markdown]
# # RSNA Pneumonia Detection Coding Assignment
# **Student Name:** ISHAAN KUMAR  
# **Registration ID:** S24BCAU0183  
# **Batch:** B6  

# %% [markdown]
# ## 1. Setup and Configuration
# Importing the modularized components (similar to the professional `afjak` codebase structure)
# and initializing the environment.

# %%
import torch
import warnings
import config
from data_preparation import prepare_data
from model import get_model
from train import train_model
from evaluate import evaluate_model
from visualize import visualize_predictions, plot_comparison

warnings.filterwarnings('ignore')
print(f"Using device: {config.DEVICE}")

# %% [markdown]
# ## 2. Data Preparation
# Loading the dataset with the optimized configuration (2000 images).

# %%
print("Preparing data loaders...")
train_loader, val_loader, train_dataset, val_dataset = prepare_data()
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# %% [markdown]
# ## 3. Model Initialization
# We instantiate both a baseline model and an improved model utilizing transfer learning 
# and custom medical anchors designed specifically for chest X-ray lesions.

# %%
print("Initializing models...")
model_baseline = get_model(num_classes=2, pretrained=False).to(config.DEVICE)
model_improved = get_model(num_classes=2, pretrained=True).to(config.DEVICE)

# %% [markdown]
# ## 4. Model Training
# Training the baseline (1 epoch) and improved models (10 epochs). 
# Note: On a CPU, the extended 10-epoch training on 200 images will take significant time.

# %%
optimizer_b = torch.optim.Adam(model_baseline.parameters(), lr=1e-4)
baseline_losses = train_model(model_baseline, train_loader, optimizer_b, config.DEVICE, config.NUM_EPOCHS_BASELINE, name=\"Baseline\")

optimizer_i = torch.optim.Adam(model_improved.parameters(), lr=1e-4)
improved_losses = train_model(model_improved, train_loader, optimizer_i, config.DEVICE, config.NUM_EPOCHS_IMPROVED, name=\"Improved\")

# %% [markdown]
# ## 5. Evaluation & Metrics
# Computing robust diagnostic metrics: Mean IoU, Median IoU, AP, and F1 Score.

# %%
print("\n" + "="*50)
print("EVALUATING BASELINE MODEL")
print("="*50)
results_baseline = evaluate_model(model_baseline, val_loader, config.DEVICE, iou_threshold=config.IOU_THRESHOLD)
print(f"Mean IoU:   {results_baseline['mean_iou']:.4f}")
print(f"Median IoU: {results_baseline['median_iou']:.4f}")
print(f"AP @0.5:    {results_baseline['ap']:.4f}")
print(f"F1 Score:   {results_baseline['f1']:.4f}")

print("\n" + "="*50)
print("EVALUATING IMPROVED MODEL")
print("="*50)
results_improved = evaluate_model(model_improved, val_loader, config.DEVICE, iou_threshold=config.IOU_THRESHOLD)
print(f"Mean IoU:   {results_improved['mean_iou']:.4f}")
print(f"Median IoU: {results_improved['median_iou']:.4f}")
print(f"AP @0.5:    {results_improved['ap']:.4f}")
print(f"F1 Score:   {results_improved['f1']:.4f}")

# %% [markdown]
# ## 6. Visualization & Comparison
# Visualizing the difference between Ground Truth (Lime) and Predictions (Magenta).

# %%
plot_comparison(results_baseline, results_improved)

# %%
visualize_predictions(model_improved, val_dataset, config.DEVICE)
