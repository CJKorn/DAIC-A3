import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

from config import *
from utils import load_trained_model, load_datasets, get_class_names

def _silence_pgd(attack):
    print("")
    if isinstance(attack, ProjectedGradientDescent):
        try:
            attack.set_params(verbose=False)
        except Exception:
            try:
                attack.verbose = False
            except Exception:
                pass

def make_classifier(adversarial=False):
    model = load_trained_model(adversarial)
    return KerasClassifier(model=model, clip_values=(0.0, 1.0), use_logits=True)

def evaluate_full_attack(classifier, attack, dataset, batch_limit=ATTACK_BATCH_LIMIT):
    clean_correct = 0
    adversarial_correct = 0
    total = 0
    if isinstance(attack, ProjectedGradientDescent):
        _silence_pgd(attack)
        total_batches = len(dataset)
        if batch_limit is not None:
            total_batches = min(total_batches, batch_limit)
        data_iter = tqdm(dataset, total=total_batches, desc="PGD batches", leave=False)
    else:
        data_iter = dataset
    for batch_index, (images, labels) in enumerate(data_iter):
        if batch_limit is not None and batch_index >= batch_limit:
            break
        x = images.numpy()
        y = labels.numpy()
        clean_predictions = np.argmax(classifier.predict(x), axis=1)
        adversarial_examples = attack.generate(x=x, y=y)
        adversarial_predictions = np.argmax(classifier.predict(adversarial_examples), axis=1)
        total += len(y)
        clean_correct += np.sum(clean_predictions == y)
        adversarial_correct += np.sum(adversarial_predictions == y)
    clean_accuracy = clean_correct / total
    adversarial_accuracy = adversarial_correct / total

    print(f"Clean accuracy: {clean_accuracy:.4f}")
    print(f"Adversarial accuracy: {adversarial_accuracy:.4f}")

def attack_one_image(classifier, attack, dataset, attack_name):
    class_names = get_class_names()
    batch_idx = np.random.randint(0, len(dataset))
    images, labels = next(iter(dataset.skip(batch_idx).take(1)))
    idx = np.random.randint(0, images.shape[0])
    image = images[idx:idx+1].numpy()
    true_label_int = labels[idx].numpy()
    true_label_text = class_names[true_label_int]
    clean_prediction = np.argmax(classifier.predict(image), axis=1)
    adversarial_example = attack.generate(x=image, y=np.array([true_label_int]))
    adversarial_prediction = np.argmax(classifier.predict(adversarial_example), axis=1)
    adv_img_array = np.clip(adversarial_example[0] * PIXEL_MAX, 0, PIXEL_MAX).astype(np.uint8)
    img = Image.fromarray(adv_img_array)
    filename = f"{attack_name}-{true_label_text}-{batch_idx}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    img.save(filepath)
    print(f"Saved adversarial image to {filepath}")
    print(f"Original prediction = {clean_prediction}")
    print(f"Adversarial Prediction = {adversarial_prediction}")

def FGSM(full, adversarial=False):
    classifier = make_classifier(adversarial)
    _, val_ds = load_datasets()
    attack = FastGradientMethod(estimator=classifier, norm=np.inf, eps=ADV_PGD_EPS / PIXEL_MAX)
    if full:
        evaluate_full_attack(classifier, attack, val_ds)
    else:
        attack_one_image(classifier, attack, val_ds, "FGSM")

def PGD(full, adversarial=False):
    classifier = make_classifier(adversarial)
    _, val_ds = load_datasets()
    attack = ProjectedGradientDescent(
        estimator=classifier,
        norm=np.inf,
        eps=ADV_PGD_EPS / PIXEL_MAX,
        eps_step=ADV_PGD_STEP / PIXEL_MAX,
        max_iter=10,
        random_eps=False,
    )
    if full:
        evaluate_full_attack(classifier, attack, val_ds)
    else:
        attack_one_image(classifier, attack, val_ds, "PGD")

def Defence():
    print("Defence")