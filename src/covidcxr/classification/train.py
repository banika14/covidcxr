from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from covidcxr.classification.models import build_classifier
from covidcxr.utils.data import ensure_dir


def read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(config_path: str | Path, architecture: str, dataset: str) -> None:
    cfg = read_yaml(config_path)
    image_size = tuple(cfg["project"]["image_size"])
    lr = float(cfg["project"]["learning_rate"])
    epochs = int(cfg["project"]["epochs"])
    batch_size = int(cfg["project"]["batch_size"])
    patience = int(cfg["project"]["patience"])

    cls_cfg = cfg["classification"]
    if dataset == "binary":
        dataset_dir = Path(cls_cfg["binary_dir"])
        num_classes = 2
    elif dataset == "multiclass":
        dataset_dir = Path(cls_cfg["multiclass_dir"])
        num_classes = 3
    elif dataset == "segmented":
        dataset_dir = Path(cls_cfg["segmented_dir"])
        num_classes = 3
    else:
        raise ValueError("dataset must be binary, multiclass, or segmented")

    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

    class_mode = "binary" if num_classes == 2 else "categorical"
    train_gen = datagen.flow_from_directory(
        dataset_dir, target_size=image_size, batch_size=batch_size, class_mode=class_mode, subset="training", shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        dataset_dir, target_size=image_size, batch_size=batch_size, class_mode=class_mode, subset="validation", shuffle=False
    )

    model = build_classifier(architecture, input_shape=(*image_size, 3), num_classes=num_classes, dropout=0.5)
    loss = "binary_crossentropy" if num_classes == 2 else "categorical_crossentropy"
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss, metrics=["accuracy"])

    out_dir = ensure_dir(Path(cls_cfg["output_dir"]) / dataset / architecture)
    callbacks = [
        ModelCheckpoint(out_dir / "best.weights.h5", monitor="val_accuracy", save_best_only=True, save_weights_only=True),
        EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_accuracy", patience=patience, factor=0.5, min_lr=1e-4),
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transfer-learning classifier for COVID-19 CXR images.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--architecture", default="resnet50")
    parser.add_argument("--dataset", choices=["binary", "multiclass", "segmented"], default="segmented")
    args = parser.parse_args()
    train(args.config, args.architecture, args.dataset)
