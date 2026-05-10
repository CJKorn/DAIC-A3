import os

IMAGE_SIZE = (224, 224)
PIXEL_MAX = 255.0
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 100
BASE_LR = 1e-3
FINETUNE_LR = 1e-4
DATASET_DIR = "Dataset"
ATTACK_BATCH_LIMIT = None
IMAGES_DIR = "images"
MODEL_NAME = "model.keras"
ADVERSERIAL_MODEL_NAME = "model_adversarial.keras"
ADV_PGD_EPS = 8.0
ADV_PGD_STEP = 2.0
ADV_PGD_ITERS = 5
ADV_MIX_RATIO = 1.0
WARMUP_EPOCHS = 3
MIX_RAMP_EPOCHS = 0
CUSTOM_MODEL = False
BACKBONE_NAME = "backbone"
do_these_need_to_be_capitalized = False

os.makedirs(IMAGES_DIR, exist_ok=True)