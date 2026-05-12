import os
import re
import keras
import tensorflow as tf
import numpy as np
import io
from PIL import Image
from scipy.ndimage import gaussian_filter

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

def feature_squeeze(x: np.ndarray):
    levels = float(2 ** SQUEEZE_BIT_DEPTH - 1)
    x = np.round(x * levels) / levels
    if x.ndim == 4:
        x = np.stack([gaussian_filter(img, sigma=[SQUEEZE_BLUR_SIGMA, SQUEEZE_BLUR_SIGMA, 0]) for img in x])
    else:
        x = gaussian_filter(x, sigma=[SQUEEZE_BLUR_SIGMA, SQUEEZE_BLUR_SIGMA, 0])
    return np.clip(x, 0.0, 1.0).astype(np.float32)
 
def jpeg_compress(x: np.ndarray):
    def _compress_one(img):
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_uint8, mode="RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    if x.ndim == 4:
        return np.stack([_compress_one(img) for img in x])
    return _compress_one(x)


def apply_defences(image_batch: np.ndarray):
    x = feature_squeeze(image_batch)
    x = jpeg_compress(x)
    return x

def _predict_probs(model: keras.Model, image_uint8: np.ndarray, shields: bool):
    x = (image_uint8.astype(np.float32) / PIXEL_MAX)[None, ...]
    if not shields:
        return softmax(model.predict(x, verbose=0)[0]).astype(np.float32)
    x = apply_defences(x)
    # probs_rs = randomised_smoothing(model, x)
    probs_wn = noisy_weights(model, x)
    return ((probs_wn) / 1.0).astype(np.float32)


def softmax(logits: np.ndarray):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)

def randomised_smoothing(model: keras.Model, image_batch: np.ndarray):
    x_tiled = np.repeat(image_batch, RS_N_SAMPLES, axis=0)
    noise = np.random.normal(0, RS_NOISE_STD, x_tiled.shape).astype(np.float32)
    x_noisy = np.clip(x_tiled + noise, 0.0, 1.0)
    logits = model.predict(x_noisy, verbose=0)
    probs = softmax(logits)
    return probs.mean(axis=0)

def noisy_weights(model: keras.Model, x: np.ndarray):
    probs_sum = np.zeros(NUM_CLASSES, dtype=np.float32)
    x_tensor  = tf.constant(x)
    original_weights = [layer.get_weights() for layer in model.layers]
    for _ in range(WN_N_PASSES):
        for layer, orig_ws in zip(model.layers, original_weights):
            if not orig_ws:
                continue
            noisy = []
            for w in orig_ws:
                layer_std = float(np.std(w)) if w.size > 1 else 1e-3
                noise = np.random.normal(0, WN_STD * layer_std, w.shape).astype(w.dtype)
                noisy.append(w + noise)
            layer.set_weights(noisy)
        logits = model(x_tensor, training=False).numpy()[0]
        probs_sum += softmax(logits)
    for layer, orig_ws in zip(model.layers, original_weights):
        if orig_ws:
            layer.set_weights(orig_ws)
    return probs_sum / WN_N_PASSES
