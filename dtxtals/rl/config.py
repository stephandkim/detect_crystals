import numpy as np


# Detector model related parameters.
DETECTOR_WEIGHT_PATH = 'detector_weights/'

# Image types
FOLDER_NAMES = {
    'rand_image_lr' : 'rand_image_lr',
    'images': 'dataset/images/',
    'annotations': 'dataset/annotations/'
}

no_plgs = {45, 70, 87, 88, 89}
FINITE_PLGS = [n for n in range(235) if n not in no_plgs]

# Angles for theta obs
ANGLES = np.arange(0, 2.25, 0.25) + 1/8

# Parameters for downsampling the original images
IMAGE_HR_SHAPE = (1920, 2448)
RATIO_HR_LR = 16
IMAGE_LR_SHAPE = (int(IMAGE_HR_SHAPE[0]/RATIO_HR_LR), int(IMAGE_HR_SHAPE[1]/RATIO_HR_LR))

# Parameters for crystal size map
NUM_PIXELS_MAX = 16
PIXEL_TYPE = {
    'background': 0,
    'valid_pixel': 2,
    'inspected': -1
}
PATCH_LR_SHAPE = (20, 20)
PATCH_HR_SHAPE = (int(PATCH_LR_SHAPE[0] * RATIO_HR_LR), int(PATCH_LR_SHAPE[1] * RATIO_HR_LR))

PATCH_VEC_LIM_MIN = (0 + PATCH_LR_SHAPE[0] / 2, 0 + PATCH_LR_SHAPE[1] / 2)
PATCH_VEC_LIM_MAX = (IMAGE_LR_SHAPE[0] - PATCH_LR_SHAPE[0] / 2, IMAGE_LR_SHAPE[1] - PATCH_LR_SHAPE[1] / 2)

# IoU threshold for deciding previously discovered or not.
IOU_NEW_CRYSTAL_THRESHOLD = 0.1

# Action mapping
ACTION_MAPPING = {
    'left': [0, (0, -1)],
    'up_left': [1, (-1, -1)],
    'up': [2, (-1, 0)],
    'up_right': [3, (-1, 1)],
    'right': [4, (0, 1)],
    'down_right': [5, (1, 1)],
    'down': [6, (1, 0)],
    'down_left': [7, (1, -1)],
    'stop': [8, (0, 0)]
}
ACTION_MAPPING_REV = {v[0]: [k, v[1]] for k, v in ACTION_MAPPING.items()}

# Potential score parameters
ALPHA_ENC = 0.01
ALPHA_UNENC = 0.01
U_PREFACTOR = 100
NO_ENCLOSED_PENALTY = 10

# Termination parameters
GRACE_PERIOD = 5
MAX_STEPS = 200
MAX_EPISODES = 200
TERMINATION_STATUS = {
    'fnd_plg': 0,
    'no_plg': 1,
    'out_of_bnd': 2,
    'TO_fnd_plg': 3,
    'TO_no_plg': 4,
    'unsrtd': None
}
TERMINATION_STATUS_REV = {v: k for k, v in TERMINATION_STATUS.items()}

# Reward amplitudes
REWARD = {
    'fnd_plg': 1,
    'no_plg': -1,
    'out_of_bnd': -1,
    'step': 0.01,
    'TO_fnd_plg': 0,
    'TO_no_plg': -1
}
NUM_PLG_ENC_MAX = 1

# Neural net parameters
PPO_NET_ARCH = [dict(pi=[16, 16], vf=[16, 16])]
