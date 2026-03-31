"""
src/utils/infer.py

CADVision inference logic.
Uses trimesh instead of Open3D for lighter deployment.

Loads trained MVCNN from local path or Hugging Face Hub,
renders 4 views from STL file, runs inference,
returns prediction and confidence.

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
    Renders 4 views of an STL file using trimesh + matplotlib.
    Returns a list of 4 PIL Images.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import trimesh
    import io

    mesh = trimesh.load(str(stl_path))

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.faces)

    if len(vertices) == 0:
        raise ValueError(f"Could not load STL file: {stl_path}")

    # normalize — center and scale to unit size (matches open3d bbox extent)
    center = vertices.mean(axis=0)
    vertices = vertices - center
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    scale = 1.0 / extent.max()
    vertices = vertices * scale

    faces = vertices[triangles]

    # use trimesh face normals directly for shading
    face_normals = np.asarray(mesh.face_normals)
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
        model_path: str | Path = None,
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
        if model_path is None:
            local_path = Path("checkpoints/best_model.pth")
            if local_path.exists():
                model_path = local_path
                print(f"model loaded from {model_path}")
            else:
                print("downloading model from Hugging Face Hub...")
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(
                    repo_id="mathewprasanth/CADvision",
                    filename="best_model.pth"
                )
                print("model downloaded from Hub")
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}. Run training first."
                )
            print(f"model loaded from {model_path}")

        model = MVCNN(num_classes=num_classes, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def predict(self, stl_path: str | Path) -> dict:
        """
        Takes an STL file path, renders 4 views, runs MVCNN inference.
        Returns prediction dict with class name, confidence, and rendered views.
        """
        stl_path = Path(stl_path)

        pil_images = render_stl(stl_path)

        tensors = [TRANSFORM(img) for img in pil_images]
        views = torch.stack(tensors)               # [4, 3, 224, 224]
        views = views.unsqueeze(0).to(self.device) # [1, 4, 3, 224, 224]

        with torch.no_grad():
            outputs = self.model(views)
            probabilities = F.softmax(outputs, dim=1)

        predicted_idx = probabilities.argmax(dim=1).item()
        confidence = probabilities[0][predicted_idx].item()
        class_name = CLASS_NAMES[predicted_idx]

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
            "rendered_views": pil_images,
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