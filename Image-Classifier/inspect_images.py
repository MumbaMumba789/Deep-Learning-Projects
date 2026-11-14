from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "train"

classes = ["books", "bags"]

fig, axes = plt.subplots(2, 5, figsize=(14, 6))

for row, class_name in enumerate(classes):

    folder = DATA_DIR / class_name
    images = list(folder.glob("*"))

    selected = random.sample(images, min(5, len(images)))

    for col, image_path in enumerate(selected):

        image = Image.open(image_path).convert("RGB")

        axes[row, col].imshow(image)
        axes[row, col].set_title(class_name)
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()