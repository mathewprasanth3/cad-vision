"""
scripts/render_dataset.py

Renders each STL file from 4 camera angles (front, side, top, isometric)
and saves them as PNG images for MVCNN training.

Uses trimesh to load and normalize meshes, matplotlib for rendering.
Matches the rendering pipeline used in HF Space inference.

Input:  data/raw/stl/<class_folder>/<model>.STL
Output: data/renders/<class_folder>/<model>_front.png
                                    <model>_side.png
                                    <model>_top.png
                                    <model>_isometric.png

Usage:
    uv run python scripts/render_dataset.py
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh


STL_DIR = Path("data/raw/stl")
RENDER_DIR = Path("data/renders")
IMAGE_SIZE = 224

CAMERA_VIEWS = {
    "front":     (0, 0),
    "side":      (0, 90),
    "top":       (90, 0),
    "isometric": (30, 45),
}


def normalize_mesh(vertices):
    center = vertices.mean(axis=0)
    vertices = vertices - center
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    scale = 1.0 / extent.max()
    vertices = vertices * scale
    return vertices


def render_stl(stl_path: Path, output_dir: Path) -> bool:
    try:
        mesh = trimesh.load(str(stl_path))

        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.faces)

        if len(vertices) == 0:
            return False

        # normalize — center and scale to unit size
        vertices = normalize_mesh(vertices)

        # build triangle face list for matplotlib
        faces = vertices[triangles]

        # use trimesh face normals directly for shading
        face_normals = np.asarray(mesh.face_normals)
        light = np.array([1, 1, 1], dtype=float)
        light = light / np.linalg.norm(light)
        brightness = np.clip(face_normals @ light, 0.2, 1.0)
        colors = plt.cm.Blues(brightness * 0.6 + 0.3)

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = stl_path.stem

        for view_name, (elev, azim) in CAMERA_VIEWS.items():
            fig = plt.figure(figsize=(IMAGE_SIZE / 100, IMAGE_SIZE / 100), dpi=100)
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

            out_path = output_dir / f"{stem}_{view_name}.png"
            plt.savefig(str(out_path), dpi=100, bbox_inches="tight",
                        facecolor="white", format="png")
            plt.close(fig)

        return True

    except Exception as e:
        print(f"failed to render {stl_path.name}: {e}")
        return False


def main() -> None:
    print("CADVision - rendering dataset with trimesh")

    if not STL_DIR.exists():
        print("STL directory not found, run download_data.py first")
        sys.exit(1)

    class_folders = sorted([f for f in STL_DIR.iterdir() if f.is_dir()])
    print(f"found {len(class_folders)} classes")

    total_success = 0
    total_failed = 0

    for class_folder in class_folders:
        stl_files = sorted(
            list(class_folder.glob("*.STL")) + list(class_folder.glob("*.stl"))
        )

        output_dir = RENDER_DIR / class_folder.name
        print(f"\nrendering {class_folder.name} ({len(stl_files)} files)")

        for stl_file in tqdm(stl_files, desc=class_folder.name, unit="model"):
            success = render_stl(stl_file, output_dir)
            if success:
                total_success += 1
            else:
                total_failed += 1

    print(f"\nrendering complete")
    print(f"success: {total_success}")
    print(f"failed:  {total_failed}")
    print(f"renders saved to {RENDER_DIR}")
    print("next: uv run python scripts/train.py")


if __name__ == "__main__":
    main()