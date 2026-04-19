# 🧩 CADVision

**Mathew Prasanth**  
AI/ML Engineer  

---

## 🚀 Overview
A deep learning system that takes an STL CAD file, renders multiple views, and automatically predicts the machining feature with visual explainability.

---

## 🔑 Features
- Input: STL file  
- Output: Feature class + confidence score  
- Grad-CAM heatmap showing prediction region  
- Multi-view (4 angles) feature recognition  

---

## 🧠 Tech Stack
Python · PyTorch · ResNet18 · MVCNN · Grad-CAM · Gradio · Hugging Face  

---

## ⚙️ Model
- **MVCNN (Multi-View CNN):** Processes 4 rendered views  
- **ResNet18:** Feature extractor (shared weights)  
- **Max Pooling:** Combines multi-view features  
- **Grad-CAM:** Visual explainability  

---

## 📊 Performance
- ~99.5% test accuracy  
- Trained on 24 classes (FeatureNet dataset)  

---

## 📂 Structure
app/ – Gradio UI  
src/ – Models & inference  
scripts/ – Data + training  
hf-space/ – Deployment  

---

## 🌐 Deployment
Hosted on **Hugging Face Spaces** with model weights on HF Hub.

---

## ⚠️ Limitations
- Works on single-feature parts only  
- Multi-feature detection not supported yet  

---
