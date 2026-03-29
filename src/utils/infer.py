"""
src/utils/infer.py

CADVision inference logic.
Mirrors the role of src/predict.py in CNC Fault Detector.

Loads trained MVCNN, renders 4 views from STL file,
runs inference, returns prediction and confidence.

Usage:
    from src.utils.infer import CADVisionPredictor
    predictor = CADVisionPredictor()
    result = predictor.predict("part.stl")
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.append(".")
from src.models.mvcnn import MVCNN


VIEWS = ["front", "side", "top", "isometric"]

CAMERA_VIEWS = {
    "front":     (0, 0),
    "side":      (0, 90),
    "top":       (90, 0),
    "isometric": (30, 45),
}

CLASS_NAMES = [
    "Oring",
    "through_hole",
    "blind_hole",
    "triangular_passage",
    "rectangular_passage",
    "circular_through_slot",
    "triangular_through_slot",
    "rectangular_through_slot",
    "rectangular_blind_slot",
    "triangular_pocket",
    "rectangular_pocket",
    "circular_end_pocket",
    "triangular_blind_step",
    "circular_blind_step",
    "rectangular_blind_step",
    "rectangular_through_step",
    "two_sides_through_step",
    "slanted_through_step",
    "chamfer",
    "round",
    "v_circular_end_blind_slot",
    "h_circular_end_blind_slot",
    "six_sides_passage",
    "six_sides_pocket",
]

IMAGE_SIZE = 224

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def render_stl(stl_path: str | Path, image_size: int = IMAGE_SIZE) -> list:
    """
    Renders 4 views of an STL file using Open3D + matplotlib.
    Returns a list of 4 PIL Images.
    Same logic as render_dataset.py — reused here for inference.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import open3d as o3d
    import io

    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Could not load STL file: {stl_path}")

    mesh.compute_vertex_normals()

    # normalize
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    mesh.translate(-center)
    extent = bbox.get_extent()
    scale = 1.0 / max(extent)
    mesh.scale(scale, center=[0, 0, 0])

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)
    faces = vertices[triangles]

    face_normals = normals[triangles].mean(axis=1)
    light = np.array([1, 1, 1], dtype=float)
    light = light / np.linalg.norm(light)
    brightness = np.clip(face_normals @ light, 0.2, 1.0)
    colors = plt.cm.Blues(brightness * 0.6 + 0.3)

    images = []
    for view_name, (elev, azim) in CAMERA_VIEWS.items():
        fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_axis_off()

        poly = Poly3DCollection(faces, facecolors=colors, linewidths=0)
        ax.add_collection3d(poly)

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(-0.6, 0.6)
        ax.view_init(elev=elev, azim=azim)
        fig.patch.set_facecolor("white")
        plt.tight_layout(pad=0)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor="white")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        images.append(img)
        plt.close(fig)

    return images


class CADVisionPredictor:

    def __init__(
        self,
        model_path: str | Path = "checkpoints/best_model.pth",
        num_classes: int = 24,
    ):
        self.device = self._get_device()
        self.model = self._load_model(model_path, num_classes)

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _load_model(self, model_path, num_classes):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run training first."
            )

        model = MVCNN(num_classes=num_classes, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        print(f"model loaded from {model_path}")
        return model

    def predict(self, stl_path: str | Path) -> dict:
        """
        Takes an STL file path, renders 4 views, runs MVCNN inference.
        Returns prediction dict with class name, confidence, and rendered views.
        """
        stl_path = Path(stl_path)

        # render 4 views from STL
        pil_images = render_stl(stl_path)

        # apply transforms and stack into tensor
        tensors = [TRANSFORM(img) for img in pil_images]
        views = torch.stack(tensors)               # [4, 3, 224, 224]
        views = views.unsqueeze(0).to(self.device) # [1, 4, 3, 224, 224]

        # inference
        with torch.no_grad():
            outputs = self.model(views)            # [1, 24]
            probabilities = F.softmax(outputs, dim=1)  # [1, 24]

        predicted_idx = probabilities.argmax(dim=1).item()
        confidence = probabilities[0][predicted_idx].item()
        class_name = CLASS_NAMES[predicted_idx]

        # top 3 predictions
        top3_probs, top3_idx = probabilities[0].topk(3)
        top3 = [
            {
                "class": CLASS_NAMES[i.item()],
                "confidence": round(p.item(), 4)
            }
            for p, i in zip(top3_probs, top3_idx)
        ]

        return {
            "predicted_class": class_name,
            "confidence": round(confidence, 4),
            "top3": top3,
            "rendered_views": pil_images,  # list of 4 PIL Images for Gradio
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: uv run python -m src.utils.infer <path_to_stl>")
        sys.exit(1)

    stl_file = sys.argv[1]
    predictor = CADVisionPredictor()
    result = predictor.predict(stl_file)

    print(f"\npredicted class: {result['predicted_class']}")
    print(f"confidence:      {result['confidence'] * 100:.2f}%")
    print(f"\ntop 3 predictions:")
    for i, pred in enumerate(result["top3"], 1):
        print(f"  {i}. {pred['class']:<30} {pred['confidence']*100:.2f}%")