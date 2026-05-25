import nbformat as nbf
import base64
import os

def embed_image_to_cell(nb, cell_idx, image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        
        output = nbf.v4.new_output(
            output_type="display_data",
            data={"image/png": img_data},
            metadata={}
        )
        nb.cells[cell_idx].outputs = [output]

# Load the existing notebook
with open("S24BCAU0183.ipynb", "r", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

# Find cells and embed images
for i, cell in enumerate(nb.cells):
    if "visualize_predictions(model_improved" in cell.source:
        embed_image_to_cell(nb, i, "predictions.png")
    elif "ax.bar(x - width/2, baseline_vals" in cell.source or "comparison.png" in cell.source:
        embed_image_to_cell(nb, i, "comparison.png")
    elif "plt.savefig('training_loss.png'" in cell.source:
        embed_image_to_cell(nb, i, "training_loss.png")
    elif "Sample Chest X-rays with Pneumonia Bounding Boxes" in cell.source:
        embed_image_to_cell(nb, i, "sample_xrays.png")

# Save the updated notebook
with open("S24BCAU0183.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook outputs successfully embedded!")
