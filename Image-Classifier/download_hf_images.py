
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import hashlib

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"

TARGET = 200


def image_signature(image):
    image = image.convert("RGB").resize((32, 32))
    return hashlib.md5(image.tobytes()).hexdigest()


def inspect_existing(folder):
    signatures = set()
    count = 0

    for file in folder.iterdir():
        try:
            with Image.open(file) as img:
                img.verify()

            with Image.open(file) as img:
                signatures.add(image_signature(img))

            count += 1

        except Exception:
            continue

    return count, signatures


def collect_images(dataset, class_name, condition):

    folder = RAW_DIR / class_name
    folder.mkdir(parents=True, exist_ok=True)

    count, signatures = inspect_existing(folder)
    needed = max(0, TARGET - count)

    print(f"\n{class_name.upper()}")
    print(f"Existing valid images: {count}")
    print(f"Images still needed: {needed}")

    if needed == 0:
        print("Target already reached!")
        return

    downloaded = 0

    for row in dataset:

        if not condition(row):
            continue

        try:
            image = row["image"].convert("RGB")

            signature = image_signature(image)

            if signature in signatures:
                continue

            filename = folder / f"hf_{class_name}_{count + downloaded:04d}.jpg"

            image.save(filename, "JPEG", quality=95)

            signatures.add(signature)
            downloaded += 1

            if downloaded % 10 == 0:
                print(f"Downloaded: {downloaded}/{needed}")

            if downloaded >= needed:
                break

        except Exception:
            continue

    print(f"New images downloaded: {downloaded}")
    print(f"Total valid images now: {count + downloaded}")


# 1. BOOK DATASET
print("\nLoading book-cover dataset...")

books_dataset = load_dataset(
    "marvy/book-covers",
    split="train",
    streaming=True
)

collect_images(
    books_dataset,
    "books",
    lambda row: "book cover" in row["prompt"].lower()
)


# 2. BAG DATASET
print("\nLoading fashion-products dataset...")

fashion_dataset = load_dataset(
    "ceyda/fashion-products-small",
    split="train",
    streaming=True
)

collect_images(
    fashion_dataset,
    "bags",
    lambda row: str(row["subCategory"]).lower() == "bags"
)

print("\nCOLLECTION COMPLETE!")
