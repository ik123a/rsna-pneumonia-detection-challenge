import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

nb_path = 'S24BCAU0183.ipynb'
print(f"Loading notebook: {nb_path}")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=14400, kernel_name='python3')
print("Executing notebook... This will take a while (CPU training).")

try:
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    print("Notebook executed successfully!")
except Exception as e:
    print(f"Error during execution: {e}")
    # Still save partial results

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Saved executed notebook: {nb_path}")
