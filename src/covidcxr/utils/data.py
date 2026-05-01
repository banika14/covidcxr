from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


def load_image(path: str | Path, image_size: tuple[int, int], grayscale: bool = False) -> np.ndarray:
    path = Path(path)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.resize(img, image_size)
    if grayscale:
        img = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return img


def list_images(folder: str | Path, suffixes: Sequence[str] = (".png", ".jpg", ".jpeg")) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    items: list[Path] = []
    for suf in suffixes:
        items.extend(folder.rglob(f"*{suf}"))
    return sorted(items)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def pair_by_stem(images: Iterable[Path], masks: Iterable[Path]) -> list[tuple[Path, Path]]:
    mask_map = {m.stem: m for m in masks}
    pairs = []
    for img in images:
        if img.stem in mask_map:
            pairs.append((img, mask_map[img.stem]))
    return pairs
