# Deep Learning Projects

A collection of small deep learning projects built with PyTorch. Each project
lives in its own folder with its own README and `requirements.txt`.

| Project | Description |
| --- | --- |
| [Image-Classifier](Image-Classifier/) | Fine-tunes a pretrained ResNet-18 to tell **books** from **bags**, with live webcam classification. |
| [Word2Vec](Word2VEc/) | Skip-gram Word2Vec trained from scratch on the text8 corpus, with embeddings visualised in the TensorBoard Projector. |

## Getting started

```powershell
git clone https://github.com/MumbaMumba789/Deep-Learning-Projects.git
cd Deep-Learning-Projects\<project-folder>
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Then follow the README in that project's folder.

## Note on large files

Datasets, model weights (`*.pt`, `*.pth`) and TensorBoard logs (`runs/`) are
not committed. Each project's README explains how to download or regenerate
them.
