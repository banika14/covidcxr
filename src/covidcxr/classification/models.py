from __future__ import annotations

from tensorflow.keras import Model
from tensorflow.keras.applications import DenseNet121, InceptionV3, ResNet50, VGG19, Xception
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input


def _base_model(name: str, input_shape: tuple[int, int, int]):
    name = name.lower()
    if name == "vgg19":
        return VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
    if name == "densenet121":
        return DenseNet121(include_top=False, weights="imagenet", input_shape=input_shape)
    if name == "resnet50":
        return ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    if name in {"inceptionv2", "inceptionv3"}:
        return InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape)
    if name == "xception":
        return Xception(include_top=False, weights="imagenet", input_shape=input_shape)
    raise ValueError(f"Unsupported architecture: {name}")


def build_classifier(name: str, input_shape=(256, 256, 3), num_classes: int = 2, dropout: float = 0.5) -> Model:
    base = _base_model(name, input_shape)
    base.trainable = False

    inputs = Input(shape=input_shape)
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)

    if num_classes == 2:
        outputs = Dense(1, activation="sigmoid")(x)
    else:
        outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name=f"{name}_classifier")
