# 🌿 TreeVisionAI — Deep Learning Tree Species Detection & Habitat Explorer

[![Model](https://img.shields.io/badge/Architecture-EfficientNet--B0-brightgreen.svg)](#model-architecture)
[![Accuracy](https://img.shields.io/badge/Val%20Accuracy-92.89%25-blue.svg)](#model-performance)
[![Classes](https://img.shields.io/badge/Classes-7%20Native%20Species-orange.svg)](#supported-species)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20FastAPI-red.svg)](#tech-stack)

**TreeVisionAI** is a deep learning computer vision system designed for high-accuracy tree species classification from leaf imagery. It features a complete end-to-end pipeline: a trained **EfficientNet-B0** model, a high-performance **FastAPI** backend, and a modern **interactive web interface** with EXIF GPS location extraction and botanical habitat insights.

---

## 🎯 Supported Tree Species

The model classifies 7 native and culturally significant tropical tree species:

| Species Key | Common Name | Scientific Name | Family |
| :--- | :--- | :--- | :--- |
| `ashoka` | False Ashoka / Mast Tree | *Polyalthia longifolia* | Annonaceae |
| `banyan` | Indian Banyan | *Ficus benghalensis* | Moraceae |
| `mango` | Mango Tree | *Mangifera indica* | Anacardiaceae |
| `neem` | Neem / Margosa | *Azadirachta indica* | Meliaceae |
| `peepal` | Sacred Fig / Peepal | *Ficus religiosa* | Moraceae |
| `raintree` | Rain Tree / Saman | *Samanea saman* | Fabaceae |
| `rubber` | Rubber Fig / Indian Rubber | *Ficus elastica* | Moraceae |

---

## ✨ Key Features

- **High-Accuracy Classification**: 92.89% validation accuracy powered by transfer learning with EfficientNet-B0.
- **Full-Stack Application**:
  - **FastAPI REST API**: Endpoints for single-image inference, top-4 ranked predictions with confidence percentages, and species metadata.
  - **Interactive Web UI**: Modern glassmorphic front-end with drag-and-drop upload, image preview, preloaded test samples, and responsive layout.
- **Botanical Knowledge Base**: Provides ecological importance, native habitat, conservation status, and leaf morphological features for every classified species.
- **EXIF GPS Extraction**: Automatically detects embedded GPS coordinates in uploaded smartphone photos and links directly to Google Maps.
- **CLI Inference**: Command-line tool to quickly test any single image from the terminal.

---

## 📁 Repository Structure

```text
├── main.py                               # FastAPI web server & UI host
├── models/
│   └── best_model.pth                    # Trained 7-class EfficientNet-B0 checkpoint (46.4 MB)
├── ui/
│   ├── index.html                        # Web application HTML interface
│   ├── style.css                         # Custom UI design system & responsive styling
│   ├── app.js                            # Frontend client logic & API integration
│   └── samples/                          # Test leaf sample images for interactive demo
│       ├── ashoka.jpg
│       ├── banyan.jpg
│       ├── mango.jpg
│       ├── neem.jpg
│       ├── peepal.jpg
│       ├── raintree.jpg
│       └── rubber.jpg
├── scripts/
│   ├── config.py                         # Single source of truth: species mapping, paths & thresholds
│   ├── model.py                          # EfficientNet-B0 architecture & checkpoint loader
│   ├── dataset.py                        # Augmentation pipelines & validation transforms
│   ├── predict.py                        # CLI prediction tool for single images
│   ├── evaluate.py                       # Evaluation script (confusion matrix & classification report)
│   ├── train.py                          # Two-phase transfer learning training pipeline
│   ├── download_species.py               # iNaturalist API dataset downloader
│   ├── clean_dataset.py                  # Automated image quality filter & deduplicator
│   ├── split_dataset.py                  # Stratified train/val/test data splitter
│   ├── verify_dataset.py                 # Dataset validation and integrity checker
│   └── generate_pdf.py                   # Technical specification PDF generator
├── outputs/
│   ├── accuracy_curve.png                # Training & validation accuracy curve
│   ├── confusion_matrix.png              # Test set confusion matrix
│   └── loss_curve.png                    # Training & validation loss curve
├── TreeVisionAI_Technical_Specifications.pdf # Complete technical specifications & architecture document
├── requirements.txt                      # Project dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+ (Python 3.10 to 3.13 supported)
- CUDA-compatible GPU (optional, runs seamlessly on CPU)

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Kishores57/species.git
cd species

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

> **Note for PyTorch GPU**: To install PyTorch with NVIDIA CUDA acceleration:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

---

## 💻 Running the Web Application

Launch the FastAPI application using Uvicorn:

```bash
uvicorn main:app --reload --port 8000
```

Open your browser and navigate to:
```
http://localhost:8000
```

1. Drag and drop any leaf image or click to select from your files.
2. Or click on any of the preloaded sample chips (Neem, Peepal, Mango, Ashoka, Banyan, Rain Tree, Rubber).
3. Click **Run Species Detection** to view predicted species, confidence scores, botanical details, and GPS location if present.

---

## 🔍 CLI Single-Image Inference

To classify an image directly from the command line:

```bash
python scripts/predict.py --image ui/samples/neem.jpg --top 3
```

Example Output:
```text
============================================================
  TreeVisionAI — Prediction Results
============================================================
Image: ui/samples/neem.jpg

  1. Neem (Azadirachta indica)
     Confidence: 99.42%

  2. Rain Tree (Samanea saman)
     Confidence: 0.35%

  3. Ashoka (Polyalthia longifolia)
     Confidence: 0.12%
============================================================
```

---

## 📡 REST API Reference

### 1. `POST /predict`
Upload a leaf image to receive classification results.
- **Request**: `multipart/form-data` with `file: <binary image>`
- **Response**:
```json
{
  "prediction": "Neem / Margosa",
  "species_key": "neem",
  "scientific_name": "Azadirachta indica",
  "confidence": "99.42%",
  "confidence_value": 99.42,
  "family": "Meliaceae (Mahogany family)",
  "found_in": "Native to the Indian Subcontinent and Southeast Asia...",
  "common_habitats": "Dry deciduous forests, arid/semi-arid plains...",
  "leaf_characteristics": "Pinnately compound leaves with 20–30 curved, sickle-shaped leaflets...",
  "ecological_importance": "Renowned as 'Nature's Pharmacy'; produces azadirachtin...",
  "conservation_status": "Least Concern (Highly Resilient)",
  "photo_gps": null,
  "top_predictions": [
    {
      "species_key": "neem",
      "display_name": "Neem / Margosa",
      "scientific_name": "Azadirachta indica",
      "confidence": 99.42
    }
  ]
}
```

### 2. `GET /api/status`
Returns server operational status, active device (CUDA / CPU), and supported classes.

### 3. `GET /api/species`
Returns the complete botanical reference database for all 7 species.

---

## 📊 Model Performance

| Metric | Value |
| :--- | :--- |
| **Validation Accuracy** | **92.89%** |
| **Training Accuracy** | **98.46%** |
| **Base Architecture** | EfficientNet-B0 (ImageNet Pretrained) |
| **Input Dimensions** | 224 × 224 pixels |
| **Training Regimen** | Phase 1 (Frozen Backbone, 10 Epochs) + Phase 2 (Fine-tuning, 30 Epochs) |

Training curves and confusion matrices are available in the [`outputs/`](outputs/) directory.

---

## 📜 License
This project is developed for educational, botanical research, and ecological conservation purposes.
