import keras
import tensorflow as tf
import numpy as np
from keras import layers
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import ProjectedGradientDescent
from tqdm import tqdm

from config import * #technically not supposed to do this, practially still am :)
from utils import get_model_path
from attack import _silence_pgd

def make_model(input_shape, num_classes):
    # data_augmentation = keras.Sequential([
    #     layers.RandomFlip("horizontal"),
    #     layers.RandomRotation(0.2),
    #     layers.RandomZoom(0.2),
    #     layers.RandomContrast(0.2),
    # ], name="data_augmentation")
    if CUSTOM_MODEL:
        inputs = keras.Input(shape=input_shape)
        # x = data_augmentation(inputs)

        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation=None)(x)
        return keras.Model(inputs, outputs)

    else:
        base = keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
        base.trainable = False
        for layer in base.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False

        inputs = keras.Input(shape=input_shape)
        x = inputs * 255.0
        # x = data_augmentation(inputs)
        x = base(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes)(x)
        return keras.Model(inputs, outputs)

class AdversarialValAccuracy(keras.callbacks.Callback): #This also sucks, but it works
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds
        self.classifier = None
        self.attack = None

    def set_model(self, model):
        super().set_model(model)
        self.classifier = KerasClassifier(model=self.model, clip_values=(0.0, 1.0), use_logits=True)
        self.attack = ProjectedGradientDescent(
            estimator=self.classifier,
            norm=np.inf,
            eps=ADV_PGD_EPS / PIXEL_MAX,
            eps_step=ADV_PGD_STEP / PIXEL_MAX,
            max_iter=ADV_PGD_ITERS,
            random_eps=False,
        )
        _silence_pgd(self.attack)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.classifier is None or self.attack is None:
            self.set_model(self.model)
        total_batches = len(self.val_ds)
        target_batches = max(1, int(total_batches * ADV_MIX_RATIO))
        selected = np.random.choice(total_batches, target_batches, replace=False)
        selected.sort()
        pbar = tqdm(total=target_batches, desc="PGD val", leave=False)
        correct = 0
        total = 0
        pick_idx = 0
        last_idx = int(selected[-1])
        for batch_index, (images, labels) in enumerate(self.val_ds):
            if batch_index > last_idx:
                break
            if batch_index != int(selected[pick_idx]):
                continue
            x = images.numpy()
            y = labels.numpy()
            adv = self.attack.generate(x=x, y=y)
            preds = np.argmax(self.classifier.predict(adv), axis=1)
            correct += np.sum(preds == y)
            total += len(y)
            pick_idx += 1
            pbar.update(1)
            if pick_idx >= target_batches:
                break
        pbar.close()
        logs["val_adv_accuracy"] = (correct / total) if total else 0.0

class AdversarialSequence(keras.utils.Sequence): #This sucks
    def __init__(self, dataset, attack, mix_ratio=0.5, target_eps=ADV_PGD_EPS, target_step=ADV_PGD_STEP):
        self.dataset = dataset
        self.attack = attack
        self.mix_ratio = mix_ratio
        self.target_eps = target_eps / PIXEL_MAX
        self.target_step = target_step / PIXEL_MAX
        self._epoch = 0
        _silence_pgd(self.attack)
        self._cached_batches = []
        total_batches = len(self.dataset)
        data_iter = tqdm(self.dataset, total=total_batches, desc="Cache clean batches", leave=False)
        for images, labels in data_iter:
            x = images.numpy().astype(np.float32)
            y = labels.numpy().astype(np.int32)
            self._cached_batches.append((x, y))
        self.augmenter = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ])

    def __len__(self):
        return len(self._cached_batches)

    def __getitem__(self, index):
        x, y = self._cached_batches[index]
        x = self.augmenter(x, training=True).numpy()
        if self._epoch < WARMUP_EPOCHS:
            return x, y
        batch_size = x.shape[0]
        if MIX_RAMP_EPOCHS > 0:
            ramp = min(1.0, (self._epoch - WARMUP_EPOCHS + 1) / MIX_RAMP_EPOCHS)
        else:
            ramp = 1.0
        effective_mix = self.mix_ratio * ramp
        adv_count = int(batch_size * effective_mix)
        if adv_count <= 0:
            return x, y
        current_eps = max(1.0 / PIXEL_MAX, self.target_eps * ramp) 
        current_step = max(0.5 / PIXEL_MAX, self.target_step * ramp)
        self.attack.set_params(eps=current_eps, eps_step=current_step)
        adv_idx = np.random.choice(batch_size, adv_count, replace=False)
        adv = self.attack.generate(x=x[adv_idx], y=y[adv_idx])
        x_mixed = x.copy()
        x_mixed[adv_idx] = adv
        return x_mixed, y

    def on_epoch_end(self):
        self._epoch += 1

def train(adversarial=False):
    if adversarial:
        print("Training adversarial model")
    else:
        print("Training model")
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    def _normalize(x, y): #WHy are there like 3 different ways to normalize
        return tf.cast(x, tf.float32) / PIXEL_MAX, y

    train_ds = train_ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    model = make_model(input_shape=IMAGE_SIZE + (3,), num_classes=NUM_CLASSES)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=BASE_LR, clipnorm=1.0),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    if adversarial:
        callbacks = [
            AdversarialValAccuracy(val_ds),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_adv_accuracy", factor=0.6, patience=5, min_lr=1e-7, mode="max", verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_adv_accuracy", patience=10, restore_best_weights=True, mode="max", verbose=1, min_delta=0.005
            )
        ]
        classifier = KerasClassifier(model=model, clip_values=(0.0, 1.0), use_logits=True)
        attack = ProjectedGradientDescent(
            estimator=classifier, norm=np.inf, eps=ADV_PGD_EPS / PIXEL_MAX,
            eps_step=ADV_PGD_STEP / PIXEL_MAX, max_iter=ADV_PGD_ITERS, random_eps=False,
        )
        _silence_pgd(attack)
        train_data = AdversarialSequence(train_ds, attack, mix_ratio=ADV_MIX_RATIO)
    else:
        train_data = train_ds.prefetch(tf.data.AUTOTUNE)
    if not CUSTOM_MODEL and WARMUP_EPOCHS > 0:
        print(f"\nWarming up for {WARMUP_EPOCHS} epochs")
        model.fit(
            train_data, 
            validation_data=val_ds, 
            epochs=WARMUP_EPOCHS, 
            callbacks=callbacks
        )
        print(f"Unfreezing")
        model.trainable = True
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=FINETUNE_LR, clipnorm=1.0),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        history = model.fit(
            train_data, 
            validation_data=val_ds, 
            epochs=EPOCHS, 
            initial_epoch=WARMUP_EPOCHS,
            callbacks=callbacks
        )
    else:
        history = model.fit(train_data, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    print(history.history["accuracy"])
    model.save(get_model_path(adversarial))