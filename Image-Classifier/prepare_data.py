
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
DATA_DIR = BASE_DIR / "data"

classes = ["books", "bags"]

for class_name in classes:

    print(f"\nPreparing {class_name}...")

    source_dir = RAW_DIR / class_name

    # Collect valid images
    images = []

    for file in source_dir.iterdir():
        try:
            with Image.open(file) as img:
                img.verify()
            images.append(file)
        except Exception:
            print(f"Skipping corrupted image: {file.name}")

    print(f"Valid images: {len(images)}")

    # Split 80/20
    train_images, test_images = train_test_split(
        images,
        test_size=0.2,
        random_state=42
    )

    # Copy images into their respective folders
    for split, split_images in [
        ("train", train_images),
        ("test", test_images)
    ]:

        destination = DATA_DIR / split / class_name
        destination.mkdir(parents=True, exist_ok=True)

        for index, image_path in enumerate(split_images):
            shutil.copy2(
                image_path,
                destination / f"{class_name}_{index}{image_path.suffix.lower()}"
            )

    print(f"Training images: {len(train_images)}")
    print(f"Testing images: {len(test_images)}")

print("\nDATA PREPARATION COMPLETE!")
