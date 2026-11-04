# Books and Bags Image Classifier

This project trains an image classifier to distinguish between **bags** and
**books**. It uses a pretrained ResNet-18 model from `torchvision`, trains a
new classification layer on the project images, and can classify images from a
webcam.

## What you need

- Python with `pip`
- A webcam to use live classification
- An internet connection the first time the pretrained ResNet-18 weights are
  needed (or when downloading images)

## Set up

Open a terminal in the project folder. On Windows PowerShell, create and
activate a virtual environment, then install the packages used by the training
and webcam scripts:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision opencv-python pillow scikit-learn tqdm tensorboard
```

If PowerShell prevents environment activation, you can run the commands using
the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision opencv-python pillow scikit-learn tqdm tensorboard
```

## Prepare the images

The training scripts expect images arranged in class-named folders:

```text
data/
  train/
    bags/
    books/
  test/
    bags/
    books/
```

To create this structure from images in `raw/books` and `raw/bags`, run:

```powershell
python prepare_data.py
```

The script skips invalid images and copies a reproducible 80/20 train/test
split into `data/train` and `data/test`. Put images you have permission to use
in the corresponding `raw` class folder before running it. Review the split
folders before rerunning preparation, since the script does not clear
previously copied files.

## Train the classifier

Run:

```powershell
python train.py
```

Training uses the `data/train` and `data/test` folders, resizes images to
224 × 224, and uses the available CUDA GPU or CPU. The current training
settings are 5 epochs, a batch size of 16, and a learning rate of 0.001;
change these constants near the top of `train.py` to adjust them.

When training finishes:

- The trained model weights are saved to `books_bags_model.pth`.
- Loss and accuracy logs are written under `runs/books_bags`.

To view the logged metrics in TensorBoard, run this in another terminal:

```powershell
tensorboard --logdir runs
```

Then open the local address printed by TensorBoard in your browser.

## Classify with your webcam

After training has produced `books_bags_model.pth`, run:

```powershell
python webcam.py
```

A webcam preview opens with the prompt **“Show me a book or a bag.”** Hold an
item in view and press **Space** to capture and classify that frame. The result
(**“IT IS A BOOK”** or **“IT IS A BAG”**) appears in large letters with the
confidence score; press **Space** again to classify a new frame. Press **Q** to
close the preview. If the webcam cannot be opened, check that it is connected,
not in use by another application, and available to your operating system.

## Other scripts

These are optional helpers, not required when you already have images:

- `download_images.py` uses Bing image search through `icrawler` to collect
  images into `raw/books` and `raw/bags`.
- `download_hf_images.py` streams selected book-cover and fashion-product
  datasets from Hugging Face into the raw folders.
- `find_datasets.py` searches Hugging Face for datasets.
- `inspect_datasets.py` displays metadata for candidate Hugging Face datasets.
- `inspect_images.py` displays a sample of the training images.

Install optional helper packages if you use these scripts:

```powershell
python -m pip install icrawler datasets huggingface_hub matplotlib
```

Downloaded images can have incorrect labels or usage restrictions. Check image
quality, class labels, and dataset terms before using them, then run
`prepare_data.py` and train again.

## Project files

- `data_setup.py` builds the training and test data loaders and image
  transforms.
- `model_builder.py` creates the pretrained ResNet-18 model and replaces its
  final layer for the project classes.
- `engine.py` implements one training epoch and one evaluation pass.
- `train.py` runs training, logs metrics, and saves the model.
- `webcam.py` loads the saved model and performs webcam inference.
- `prepare_data.py` validates raw images and copies the train/test split.
