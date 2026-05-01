from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from covidcxr.utils.data import ensure_dir, list_images


def add_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(mask, 0.3, 1, cv2.THRESH_BINARY)
    merged = cv2.addWeighted(img, 0.7, mask.astype(np.float64), 0.3, 0)
    return merged


def segment_folder(model_path: str | Path, input_dir: str | Path, output_dir: str | Path, image_size=(256, 256)) -> None:
    model = load_model(model_path, compile=False)
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)
    kernel = np.ones((5, 5), np.uint8)

    for img_path in list_images(input_dir):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        resized = cv2.resize(img, image_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        pred = model.predict(np.expand_dims(np.expand_dims(gray, axis=-1), axis=0), verbose=0)[0, ..., 0]
        eroded = cv2.erode(pred.astype("float32"), kernel, iterations=2)
        merged = add_mask(gray, eroded)
        out_path = output_dir / f"{img_path.stem}_segmented.png"
        cv2.imwrite(str(out_path), (merged * 255).astype("uint8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply the trained lung segmentation model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    segment_folder(args.model, args.input_dir, args.output_dir)
