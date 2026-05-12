import os
import numpy as np
from pick import pick
from inference_view import show_inference_window

from config import IMAGES_DIR
from train import train
from attack import FGSM, PGD
from utils import (
    get_class_names, _derive_true_class_from_filename, _load_image_uint8, 
    _predict_probs, load_trained_model
)

def main():
    while True:
        option, index = pick(['Train', 'Attack Full', 'Attack One', 'Test'], "Whatcha wanna do?")
        match option:
            case 'Train':
                adversarial = adv_or_no("Pick training mode")
                train(adversarial)
            case 'Attack Full':
                adversarial = adv_or_no("Pick model to attack")
                shields = Shields()
                option, index = pick(['FGSM', 'PGD'], "Pick an attack method")
                match option:
                    case 'FGSM':
                        FGSM(True, adversarial, shields=shields)
                    case 'PGD':
                        PGD(True, adversarial, shields=shields)
            case 'Attack One':
                adversarial = adv_or_no("Pick model to attack")
                shields = Shields()
                option, index = pick(['FGSM', 'PGD'], "Pick an attack method")
                match option:
                    case 'FGSM':
                        FGSM(False, adversarial, shields=shields)
                    case 'PGD':
                        PGD(False, adversarial, shields=shields)
            case 'Test':
                adversarial = adv_or_no("Pick model to test")
                shields = Shields()
                Test(adversarial=adversarial, shields=shields)
            case _:
                print("Literally how did you get here")
                return
        input("Press enter to go back to the menu thingy ⊂(◉‿◉)つ")

def adv_or_no(prompt):
    option, _ = pick(["Normal", "Adversarially trained"], prompt)
    return option == "Adversarially trained"

def Shields():
    option, _ = pick(["No", "Yes"], "Divert power to shields?")
    return option == "Yes"

def Test(shields: bool | None = None, adversarial=False):
    image_files = sorted(
        [
            f
            for f in os.listdir(IMAGES_DIR)
            if os.path.isfile(os.path.join(IMAGES_DIR, f))
            and os.path.splitext(f.lower())[1] in {".png", ".jpg", ".jpeg", ".webp"}
        ]
    )

    if not image_files:
        print(f"No images found in '{IMAGES_DIR}/'")
        return
    picked_file, _ = pick(image_files, "Pick an image")
    image_path = os.path.join(IMAGES_DIR, picked_file)
    if shields is None:
        mode, _ = pick(["Base Model", "Base + Defences"], "Divert power to shields?")
        shields = mode == "Base + Defences"
    class_names = get_class_names()
    correct_class = _derive_true_class_from_filename(picked_file, class_names)
    image_uint8 = _load_image_uint8(image_path)
    model = load_trained_model(adversarial)
    probs = _predict_probs(model, image_uint8, shields)
    pred_idx = int(np.argmax(probs))
    predicted_class = class_names[pred_idx]
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5 = [(class_names[int(i)], float(probs[int(i)])) for i in top5_idx]

    show_inference_window(
        image_uint8=image_uint8,
        predicted_class=predicted_class,
        top5=top5,
        correct_class=correct_class,
        window_title=picked_file,
    )

if __name__ == "__main__":
    main()