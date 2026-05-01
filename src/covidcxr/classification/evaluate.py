from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model


def evaluate_model(model_path: str | Path, generator) -> dict:
    model = load_model(model_path, compile=False)
    y_true = generator.classes
    preds = model.predict(generator)
    if preds.shape[-1] == 1:
        y_pred = (preds.ravel() >= 0.5).astype(int)
        average = "binary"
    else:
        y_pred = np.argmax(preds, axis=1)
        average = "macro"

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }
