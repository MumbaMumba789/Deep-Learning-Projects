# Skip-gram Word2Vec on text8

A from-scratch PyTorch implementation of the **skip-gram Word2Vec** model,
trained on the [text8](http://mattmahoney.net/dc/textdata.html) corpus (the
first 100 MB of cleaned English Wikipedia). The model learns a dense vector for
each word so that words appearing in similar contexts end up close together in
embedding space. Training curves and the learned embeddings can be explored in
TensorBoard, including an interactive 3-D view in the Embedding Projector.

Everything lives in a single notebook: `skipgram_word2vec.ipynb`.

## Set up

From this folder, on Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch that matches your system
from <https://pytorch.org/get-started/locally/> *before* installing the
remaining requirements.

## Run

```powershell
jupyter notebook skipgram_word2vec.ipynb
```

Or open the notebook in VS Code and select the `.venv` kernel. Run the cells
top to bottom:

1. **Data.** The notebook downloads `text8.zip` into `data/` and extracts it.
   This only happens the first time; afterwards the local copy is reused.
2. **Vocabulary.** Builds word counts and the word-to-index mapping.
3. **Training.** Trains the skip-gram model and logs the loss to
   `runs/skipgram_text8/`.
4. **Saving.** Writes the trained weights to `skipgram_text8.pt`.
5. **Embeddings.** Exports the vectors for the most frequent words, with their
   labels, to `runs/skipgram_text8_projector/` for the TensorBoard projector.

## View results in TensorBoard

```powershell
tensorboard --logdir runs
```

Open the address that TensorBoard prints (usually <http://localhost:6006>).

- **Scalars** shows the training loss curve.
- **Projector** shows the word embeddings in 3-D with PCA, t-SNE or UMAP. Search
  for a word to see its nearest neighbours.

## Files not stored in Git

The dataset, checkpoints and logs are large, and the notebook regenerates them,
so `.gitignore` excludes them:

| Path | What it is |
| --- | --- |
| `data/text8`, `data/text8.zip` | text8 corpus (~100 MB), downloaded by the notebook |
| `*.pt` | Trained model checkpoints |
| `runs/` | TensorBoard logs and projector exports |
