# 🔩 CADVision

Multi-view deep learning system that takes an STL CAD file, renders it from 4 angles, and automatically predicts the machining feature class — with visual explainability via Grad-CAM.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-yellow)](https://huggingface.co/spaces/mathewprasanth/CADVision)
[![Model Weights](https://img.shields.io/badge/Weights-HuggingFace-blue)](https://huggingface.co/mathewprasanth/CADVisionWeights)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML-red)](https://pytorch.org)

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | 99.56% |
| Classes | 24 machining features |
| Views per sample | 4 rendered angles |
| Dataset | FeatureNet |

---

## 🧠 What It Does

Identifying machining features from CAD geometry is a time-consuming step in manufacturing process planning. This system automates feature recognition from raw STL files, enabling faster downstream decisions around toolpath selection, fixturing, and setup planning.

The pipeline performs three tasks:
- Renders the STL file from 4 different viewing angles
- Classifies the machining feature using a Multi-View CNN with shared ResNet18 encoders
- Generates Grad-CAM heatmaps showing which regions of the geometry drove the prediction

This reflects real-world CAM workflows where feature recognition is the first step before toolpath generation.

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Rendering | Multi-view STL renderer (4 angles) |
| Feature Extractor | ResNet18 (shared weights across views) |
| Fusion | MVCNN with max pooling across views |
| Explainability | Grad-CAM (custom PyTorch hooks) |
| Backend | PyTorch |
| UI | Gradio |
| Hosting | Hugging Face Spaces |
| Model Storage | Hugging Face Model Hub |

---

## 🏗️ Pipeline Architecture

```
STL File
→ Multi-view renderer (4 angles)
→ ResNet18 encoder (shared weights, one per view)
→ Max pooling (fuse 4 view features)
→ Classification head (24 classes)
→ Grad-CAM (attention heatmap per view)
→ Output: feature class + confidence + explainability
```

---

## 🔑 Key Training Decisions

- Shared ResNet18 weights across views — view-invariant feature learning with fewer parameters
- Max pooling over mean pooling — robust to uninformative views, captures strongest activations
- Grad-CAM on individual views — shows which geometry region drove the prediction
- Device-agnostic design (CPU / MPS / CUDA)

---

## 📁 Project Structure

```
cadVision/
├── app/
│   └── app.py
├── requirements.txt
├── src/
│   ├── models/
│   └── inference/
├── scripts/
│   ├── data/
│   └── train/
└── hf-space/
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/mathewprasanth/cadVision.git
cd cadVision
pip install -r requirements.txt
python app/app.py
```

## Training

```bash
python scripts/train/train.py
```

---

## ⚠️ Limitations

- Works on single-feature parts only
- Multi-feature detection not yet supported

---

## 👤 Author

**Mathew Prasanth, P.E.**
AI/ML Engineer | U.S. Licensed Professional Engineer
[LinkedIn](https://www.linkedin.com/in/mathewprasanth/) · [Live Demo](https://huggingface.co/spaces/mathewprasanth/CADVision)

*AWS Certified ML Specialty · AWS Cloud Practitioner*
