import os

IMAGE_SIZE = (224, 224)
PIXEL_MAX = 255.0
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 100
BASE_LR = 5e-4
FINETUNE_LR = 3e-4
DATASET_DIR = "Dataset"
ATTACK_BATCH_LIMIT = None
IMAGES_DIR = "images"
MODEL_NAME = "model.keras"
ADVERSERIAL_MODEL_NAME = "model_adversarial.keras"
ADV_PGD_EPS = 4.0
ADV_PGD_STEP = 2.0
ADV_PGD_ITERS = 3
ADV_MIX_RATIO = 0.4
WARMUP_EPOCHS = 10
MIX_RAMP_EPOCHS = 15
CUSTOM_MODEL = False
BACKBONE_NAME = "backbone"
SQUEEZE_BIT_DEPTH   = 8
SQUEEZE_BLUR_SIGMA  = 0.5
JPEG_QUALITY        = 85
RS_N_SAMPLES        = 32
RS_NOISE_STD        = 0.03
NOISE_AWARE_TRAINING = True
MCD_N_SAMPLES       = 32
WN_STD              = 1e-4
WN_N_PASSES         = 8

do_these_need_to_be_capitalized = False

os.makedirs(IMAGES_DIR, exist_ok=True)