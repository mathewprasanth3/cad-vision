# CADVision

Multi-view deep learning system that takes an STL CAD file, renders it from 4 angles, and automatically predicts the machining feature class — with visual explainability via Grad-CAM.

**Live Demo** → https://huggingface.co/spaces/mathewprasanth/CADVision  
**Model Weights** → https://huggingface.co/mathewprasanth/CADVisionWeights

---

## What It Does

Identifying machining features from CAD geometry is a time-consuming step in manufacturing process planning. This system automates feature recognition from raw STL files, enabling faster downstream decisions around toolpath selection, fixturing, and setup planning.

The pipeline performs three tasks:
- Renders the STL file from 4 different viewing angles
- Classifies the machining feature using a Multi-View CNN with shared ResNet18 encoders
- Generates Grad-CAM heatmaps showing which regions of the geometry drove the prediction

This reflects real-world CAM workflows where feature recognition is the first step before toolpath generation.

---

## Tech Stack

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

## Dataset

**FeatureNet Dataset** — 24 machining feature classes.

| Property | Value |
|---|---|
| Classes | 24 machining features |
| Input | STL files |
| Views per file | 4 rendered angles |

---

## Pipeline Architecture

```
STL File
-> Multi-view renderer (4 angles)
-> ResNet18 encoder (shared weights, one per view)
-> Max pooling (fuse 4 view features)
-> Classification head (24 classes)
-> Grad-CAM (attention heatmap per view)
-> Output: feature class + confidence + explainability
```

---

## Model Architecture

### MVCNN (Multi-View CNN)
- Encoder: ResNet18 with shared weights across all 4 views
- Fusion: Max pooling across view embeddings
- Classifier: Fully connected head, 24 output classes

### Performance
- Test accuracy: ~99.5%
- Classes: 24 machining features (FeatureNet dataset)

---

## Key Training Decisions

- Shared ResNet18 weights across views — view-invariant feature learning
- Max pooling fusion — robust to view ordering, captures strongest activations
- Grad-CAM on individual views — shows which geometry region drove the prediction
- Device-agnostic design (CPU / MPS / CUDA)

---

## Project Structure

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

## Inference Pipeline

1. Upload STL file
2. Renderer produces 4 views at different angles
3. ResNet18 encodes each view independently with shared weights
4. Max pooling fuses 4 view embeddings into one feature vector
5. Classifier predicts machining feature class and confidence
6. Grad-CAM generates heatmap showing prediction region on each view

---

## Deployment

- Hosted on Hugging Face Spaces
- Model weights stored on Hugging Face Model Hub
- Downloaded dynamically at runtime
- Runs on CPU (HF Spaces constraint)

---

## Key Engineering Decisions

- Shared encoder weights across views — one set of parameters learns view-invariant geometry features
- Max pooling over mean pooling — more robust to uninformative views
- Models loaded once at startup for performance
- Clear separation between rendering, inference, and UI

---

## Run Locally

```bash
git clone https://github.com/mathewprasanth3/cadVision.git
cd cadVision
pip install -r requirements.txt
python app/app.py
```

---

## Training

```bash
python scripts/train/train.py
```

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | ~99.5% |
| Classes | 24 |
| Views per sample | 4 |

---

## Limitations

- Works on single-feature parts only
- Multi-feature detection not supported yet

---

## Author

**Mathew Prasanth, PE**  
AI/ML Engineer  
[https://www.linkedin.com/in/mathewprasanth/](https://www.linkedin.com/in/mathewprasanth/)  
[https://huggingface.co/spaces/mathewprasanth/CADVision](https://huggingface.co/spaces/mathewprasanth/CADVision)

AWS Certified Cloud Practitioner · AWS Certified Machine Learning Specialty
