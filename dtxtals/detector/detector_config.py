import albumentations.pytorch
import albumentations


MIN_PIXELS = 100
BUMPER = 1

# Training parameters
NUM_EPOCHS = 100_000
MODEL_SAVE_FREQ = 1_000
LEARNING_RATE = 0.0005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

PREDICTION_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

PIXEL_VALS = {'background': 0,
              'unfilled': 1,
              'filled': 2
              }

CRYSTAL_TYPE = {'good': 0}

CRYSTAL_TYPE_REV = {v: k for k, v in CRYSTAL_TYPE.items()}

TRANSFORM_NO_CROP = albumentations.Compose(
    [albumentations.pytorch.transforms.ToTensorV2()],
    bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']),
)

TRANSFORM_CROP = albumentations.Compose(
    [albumentations.RandomCrop(320, 320),
     albumentations.pytorch.transforms.ToTensorV2()],
    bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']),
)
