from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NATURAL_SPLIT_RE = re.compile(r"(\d+)")

# PyCharm direct-run defaults.
DEFAULT_ROOT = Path(r"F:\Microalgae_Photoes\20260520")
DEFAULT_TOTAL_IMAGES = 250
DEFAULT_OUT = Path(r"F:\Microalgae_Photoes\20260520\selected_250")


def natural_key(text: str):
    parts = NATURAL_SPLIT_RE.split(text)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def path_sort_key(path: Path, root: Path):
    rel = path.relative_to(root)
    return tuple(natural_key(part) for part in rel.parts)


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def collect_images(root: Path, out_dir: Path | None = None) -> list[Path]:
    images = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if out_dir is not None and is_within(path, out_dir):
            continue
        images.append(path)
    return sorted(images, key=lambda p: path_sort_key(p, root))


def sample_images(images: list[Path], count: int, seed: int | None) -> list[Path]:
    if count <= 0:
        return []
    if len(images) <= count:
        return list(images)
    rng = random.Random(seed)
    return rng.sample(images, count)


def resolve_output_dir(root: Path, count: int, out_dir: Path | None) -> Path:
    if out_dir is not None:
        return out_dir.resolve()
    return DEFAULT_OUT.resolve()


def make_flat_name(src: Path, rank: int, total: int) -> str:
    width = max(4, len(str(total)))
    return f"{rank:0{width}d}_{src.name}"


def build_selection_rows(selected: list[Path], root: Path, out_dir: Path):
    rows = []
    total = len(selected)
    for rank, src in enumerate(selected, start=1):
        rel = src.relative_to(root)
        flat_name = make_flat_name(src, rank, total)
        rows.append(
            {
                "sample_rank": rank,
                "source_path": str(src),
                "relative_path": str(rel),
                "flat_name": flat_name,
                "destination_path": str(out_dir / flat_name),
            }
        )
    return rows


def prepare_output_dir(out_dir: Path, root: Path):
    if out_dir.resolve() == root.resolve():
        raise SystemExit("Output folder must not be the same as the source folder.")
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in list(out_dir.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def copy_selected(rows: list[dict], out_dir: Path):
    for row in rows:
        src = Path(row["source_path"])
        dst = out_dir / row["flat_name"]
        shutil.copy2(src, dst)


def write_manifest(rows: list[dict], out_dir: Path):
    manifest_path = out_dir / "selection_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_rank",
                "source_path",
                "relative_path",
                "flat_name",
                "destination_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def write_summary(root: Path, out_dir: Path, total_found: int, total_selected: int, seed: int | None):
    summary_path = out_dir / "selection_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"root={root}\n")
        f.write(f"out={out_dir}\n")
        f.write(f"found_images={total_found}\n")
        f.write(f"selected_images={total_selected}\n")
        f.write(f"seed={seed}\n")
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(description="Randomly sample images from a folder.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Source folder that contains images. Default: F:\\Microalgae_Photoes\\20260520",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_TOTAL_IMAGES,
        help="How many images to sample. Default: 200",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output folder. Default: F:\\Microalgae_Photoes\\20260520_selected_200",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only write the manifest and summary, do not copy images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root folder not found: {root}")

    if args.count < 0:
        raise SystemExit("count must be >= 0")

    out_dir = resolve_output_dir(root, args.count, args.out)

    images = collect_images(root, out_dir=out_dir)
    if not images:
        raise SystemExit("No images found.")

    selected = sample_images(images, args.count, args.seed)
    if not args.manifest_only:
        prepare_output_dir(out_dir, root)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    selection_rows = build_selection_rows(selected, root, out_dir)

    manifest_path = write_manifest(selection_rows, out_dir)
    summary_path = write_summary(root, out_dir, len(images), len(selected), args.seed)

    if not args.manifest_only:
        copy_selected(selection_rows, out_dir)

    print(f"Found images: {len(images)}")
    print(f"Selected images: {len(selected)}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
