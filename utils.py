import os
import re
import keras
import tensorflow as tf
import numpy as np
from PIL import Image

from config import *

def load_datasets():
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    def _normalize(x, y):
        return tf.cast(x, tf.float32) / PIXEL_MAX, y

    train_ds = train_ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

def get_model_path(adversarial):
    return ADVERSERIAL_MODEL_NAME if adversarial else MODEL_NAME

def load_trained_model(adversarial=False):
    model_path = get_model_path(adversarial)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Train the model first my guy")
    return keras.models.load_model(model_path)

def get_class_names():
    return sorted([d for d in os.listdir(DATASET_DIR) 
                   if os.path.isdir(os.path.join(DATASET_DIR, d))])

def _load_image_uint8(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    target_size = (IMAGE_SIZE[1], IMAGE_SIZE[0])
    if img.size != target_size:
        img = img.resize(target_size, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)

def _derive_true_class_from_filename(filename: str, class_names: list[str]) -> str | None:
    stem = os.path.splitext(os.path.basename(filename))[0]
    tokens = [t for t in re.split(r"[-_]+", stem) if t]
    for token in tokens:
        if token in class_names:
            return token
    matches = [c for c in class_names if c in stem]
    if len(matches) == 1:
        return matches[0]
    return None

def _apply_defences(image_batch: np.ndarray) -> np.ndarray:
    return image_batch

def _predict_probs(model: keras.Model, image_uint8: np.ndarray, shields: bool) -> np.ndarray:
    x = (image_uint8.astype(np.float32) / PIXEL_MAX)[None, ...]
    if shields:
        x = _apply_defences(x)
    probs = model.predict(x, verbose=0)[0]
    return probs.astype(np.float32)