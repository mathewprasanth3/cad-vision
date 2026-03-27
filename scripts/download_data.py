"""
scripts/download_data.py

Downloads and extracts the FeatureNet machining feature dataset.
24 classes, 1000 STL models per class, 24000 total.

Usage:
    uv run python scripts/download_data.py
"""

import sys
import shutil
import subprocess
import urllib.request
from pathlib import Path


DATASET_URL = (
    "https://github.com/madlabub/Machining-feature-dataset"
    "/blob/master/dataset.rar?raw=true"
)

RAW_DIR = Path("data/raw")
RAR_PATH = RAW_DIR / "dataset.rar"
STL_DIR = RAW_DIR / "stl"


def download_with_progress(url: str, dest: Path) -> None:
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100 / total_size, 100)
            bar = "█" * int(percent / 2) + "░" * (50 - int(percent / 2))
            print(f"\r  [{bar}] {percent:.1f}%", end="", flush=True)

    print(f"downloading from {url}\n")
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()


def check_unar() -> bool:
    try:
        subprocess.run(["unar", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


def extract_rar(rar_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting {rar_path.name}...")

    temp_dir = rar_path.parent / "temp"
    temp_dir.mkdir(exist_ok=True)

    result = subprocess.run(
        ["unar", "-o", str(temp_dir), str(rar_path)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"extraction failed: {result.stderr}")
        shutil.rmtree(temp_dir)
        sys.exit(1)

    # move contents up, skipping any nested stl folder
    nested = temp_dir / "stl"
    source = nested if nested.exists() else temp_dir

    for item in source.iterdir():
        item.rename(output_dir / item.name)

    shutil.rmtree(temp_dir)
    print("extraction complete")


def verify_dataset(stl_dir: Path) -> tuple[int, int]:
    class_folders = [f for f in stl_dir.iterdir() if f.is_dir()]
    total = sum(
        len(list(f.glob("*.stl"))) + len(list(f.glob("*.STL")))
        for f in class_folders
    )
    return len(class_folders), total


def main() -> None:
    print("CADVision - downloading FeatureNet dataset")

    if not check_unar():
        print("unar is not installed. please run: brew install unar")
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # download
    if RAR_PATH.exists():
        print("dataset.rar already exists, skipping download")
    else:
        print("downloading dataset...")
        try:
            download_with_progress(DATASET_URL, RAR_PATH)
            size_mb = RAR_PATH.stat().st_size / (1024 * 1024)
            print(f"downloaded ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"download failed: {e}")
            sys.exit(1)

    # extract
    if STL_DIR.exists() and any(STL_DIR.iterdir()):
        print("already extracted, skipping")
    else:
        extract_rar(RAR_PATH, STL_DIR)

    # verify
    print("verifying dataset...")
    num_classes, total_files = verify_dataset(STL_DIR)

    if total_files == 0:
        print("verification failed, no STL files found")
        sys.exit(1)

    print(f"{num_classes} classes, {total_files} total STL files")
    print(f"dataset ready at {STL_DIR}")
    print("next: uv run python scripts/render_dataset.py")


if __name__ == "__main__":
    main()