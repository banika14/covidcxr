from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from covidcxr.segmentation.unet import build_unet
from covidcxr.utils.data import ensure_dir, list_images, load_image, pair_by_stem
from covidcxr.utils.plotting import plot_history


def read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_segmentation_data(image_dir: Path, mask_dir: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    images = list_images(image_dir)
    masks = list_images(mask_dir)
    pairs = pair_by_stem(images, masks)
    if not pairs:
        raise ValueError(f"No image/mask pairs found in {image_dir} and {mask_dir}")

    x = []
    y = []
    for img_path, mask_path in pairs:
        x.append(load_image(img_path, image_size, grayscale=True))
        y.append(load_image(mask_path, image_size, grayscale=True))
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def train(config_path: str | Path) -> None:
    cfg = read_yaml(config_path)
    image_size = tuple(cfg["project"]["image_size"])
    lr = float(cfg["project"]["learning_rate"])
    epochs = int(cfg["project"]["epochs"])
    batch_size = int(cfg["project"]["batch_size"])
    patience = int(cfg["project"]["patience"])

    seg_cfg = cfg["segmentation"]
    x, y = load_segmentation_data(Path(seg_cfg["input_dir"]), Path(seg_cfg["mask_dir"]), image_size)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    model = build_unet((*image_size, 1))
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["binary_accuracy"])

    model_path = Path(seg_cfg["weights_path"])
    ensure_dir(model_path.parent)

    callbacks = [
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, save_weights_only=True),
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=patience, factor=0.5, min_lr=1e-4),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
    )

    plot_history(history, output_path=model_path.with_name("unet_history.png"))
    model.save(model_path.with_suffix(".keras"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the U-Net lung segmentation model.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
