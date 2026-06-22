# Face Recognition

Compare facial similarity between two photos using a self-trained PCA-based model.

Built with Streamlit, OpenCV, and scikit-learn.

---

## Overview

Train a face similarity model from your own photo dataset, then compare two faces and get a scored verdict. The model can be saved as `.pkl` and reloaded anytime without retraining.

Face detection and alignment are handled automatically before comparison, so slightly tilted or off-angle shots still work.

---

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Dataset Format

Two structures are auto-detected:

**Folder per person** _(recommended)_

```
dataset.zip
├── putri/
│   ├── photo1.jpg
│   └── photo2.jpg
└── dimas/
    └── any_name.png
```

**Flat files**

```
dataset.zip
├── putri_1.jpg
├── putri_2.jpg
└── dimas_1.jpg
```

> Supported archives: `.zip`, `.7z` — Supported images: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`  
> Minimum: 2 different people, 3 photos total

---

## Scoring

The final similarity score is a weighted combination of three signals:

- **Cosine similarity** — angle between face vectors in PCA space
- **Euclidean similarity** — normalized distance in PCA space
- **SSIM** — structural pixel similarity between aligned faces

---

## Auto-Load Model (Optional)

Set `MODEL_URL` in Streamlit Secrets to load a pre-trained model on startup:

```toml
MODEL_URL = "https://raw.githubusercontent.com/username/repo/main/model.pkl"
```

---

## Project Structure

```
.
├── app.py
├── model.pkl          # generated after training
├── requirements.txt
└── .streamlit/
    └── config.toml
```

---
