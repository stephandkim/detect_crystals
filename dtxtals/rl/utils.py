import cv2
import dtxtals.rl.config as config
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import torch.nn as nn
import torchvision
import dtxtals.detector.detector_config as detector_config
import dtxtals.detector.detector_utils as detector_utils
import os
import math
from collections import namedtuple
from collections import defaultdict


Vector = namedtuple('Vector', ('y', 'x'))

class Polygon(object):
    def __init__(self):
        self.num_pixels = 0
        self.loc = set()
        self.center_of_mass = None


def create_image_lr(image_raw) -> np.array:
    # Use the standard k-means clustering algorithm to distinguish the background pixels.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts = 10
    ret, labels, center = cv2.kmeans(
                        np.float32(image_raw.reshape((-1, 3))),
                        K,
                        None,
                        criteria,
                        attempts,
                        cv2.KMEANS_PP_CENTERS
                        )
    elements, frequency = np.unique(labels, return_counts=True)
    if frequency[0] > frequency[1]:  # background is label 0
        labels = labels * config.PIXEL_TYPE['valid_pixel']
    else:  # background is label 1
        labels = (labels - 1) * -1 * config.PIXEL_TYPE['valid_pixel']

    image_bin = labels.astype(np.float32).reshape(image_raw.shape[0], image_raw.shape[1])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    t = torch.as_tensor(image_bin).unsqueeze(0).to(device)

    pad_layer = nn.ConstantPad2d(1, 0)
    max_pool = nn.MaxPool2d(config.RATIO_HR_LR, stride=config.RATIO_HR_LR)

    t = pad_layer(t)
    t = max_pool(t)

    return t.squeeze(0).to(torch.device('cpu')).numpy()


def flood_fill(image_lr, r, c):
    # Simple flood fill algorithm for counting the number of polygons.
    p = Polygon()

    queue = deque([])
    queue.append((r, c))
    center_of_mass = [0, 0]

    while queue:
        r, c = queue.popleft()

        if r < 0 or r >= config.IMAGE_LR_SHAPE[0] or \
            c < 0 or c >= config.IMAGE_LR_SHAPE[1] or \
            (r, c) in p.loc or \
            image_lr[r][c] != config.PIXEL_TYPE['valid_pixel']:
            continue
        else:
            p.loc.add(Vector(r, c))
            center_of_mass[0] += r
            center_of_mass[1] += c
            p.num_pixels += 1
            image_lr[r][c] = config.PIXEL_TYPE['inspected']

            for (j, i) in [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]:
                queue.append((r + j, c + i))

    # Update the center of mass
    if p.num_pixels > 0:
        # polygon.center_of_mass = [n/polygon.num_pixels for n in polygon.center_of_mass]
        center_of_mass[0] /= p.num_pixels
        center_of_mass[1] /= p.num_pixels
        p.center_of_mass = Vector(*center_of_mass)
    return p


def calculate_prob(num_pixels):
    # Calculate p according to the number of pixels.
    return max(1 - num_pixels/config.NUM_PIXELS_MAX, 0)


def count_polygons(image_lr):
    plgs = set()
    for r in range(config.IMAGE_LR_SHAPE[0]):
        for c in range(config.IMAGE_LR_SHAPE[1]):
            p = flood_fill(image_lr, r, c)
            if p.num_pixels > 0:
                if p.num_pixels < config.NUM_PIXELS_MAX:
                    # Only the polygons with pixels less than the threshold are of interest.
                    plgs.add(p)
                for loc in p.loc:
                    image_lr[loc.y][loc.x] = int(p.num_pixels < config.NUM_PIXELS_MAX)

    return plgs


def sort_polygons(patch_vec, polygons):
    patch_ymin, patch_ymax = patch_vec.y - config.PATCH_LR_SHAPE[0] / 2, patch_vec.y + config.PATCH_LR_SHAPE[0] / 2
    patch_xmin, patch_xmax = patch_vec.x - config.PATCH_LR_SHAPE[1] / 2, patch_vec.x + config.PATCH_LR_SHAPE[1] / 2

    enc, unenc = deque([]), deque([])  # enclosed and unenclosed polygons
    for polygon in polygons:
        unenclosed_flag = False
        # All pixels of a polygon should be within the patch for the polygon to be considered enclosed.
        for loc in polygon.loc:
            if (patch_ymin <= loc.y) and (loc.y < patch_ymax) and (patch_xmin <= loc.x) and (loc.x < patch_xmax):
                continue
            else:
                unenclosed_flag = True
                break

        if unenclosed_flag:
            unenc.append(polygon)
        else:
            enc.append(polygon)

    return enc, unenc


def calculate_l2_norm(vec1: Vector, vec2: Vector):
    return math.sqrt((vec1.y - vec2.y) ** 2 + (vec1.x - vec2.x) ** 2)


def get_closest_unenclosed_polygons(plgs_unenc, patch_vec):
    if not plgs_unenc:
        return None
    p_closest, l2_closest = None, None
    for p in plgs_unenc:
        if not l2_closest:
            p_closest, l2_closest = p, calculate_l2_norm(p.center_of_mass, patch_vec)
        else:
            l2 = calculate_l2_norm(p.center_of_mass, patch_vec)
            if l2 < l2_closest:
                p_closest, l2_closest = p, l2
    return p_closest


def get_theta_obs(plgs: deque[Polygon], patch_vec: Vector, plgs_type: str) -> np.array:
    obs_size = 8 if plgs_type == 'unenclosed' else 9
    theta_obs = np.zeros(obs_size, dtype=np.float32)

    if len(plgs) == 1 and plgs[0] is None:
        return theta_obs

    for p in plgs:
        r = Vector(p.center_of_mass.y - patch_vec.y, p.center_of_mass.x - patch_vec.x)
        theta = None

        if r.y != 0 and r.x != 0:
            theta = math.atan(r.y/r.x)/math.pi
            if r.x < 0:
                theta += 1
            elif r.x > 0 and r.y < 0:
                theta += 2
            for idx, angle in enumerate(config.ANGLES):
                if theta <= angle:
                    if idx == len(config.ANGLES) - 1:
                        theta_obs[0] = 1
                    else:
                        theta_obs[idx] = 1
                    break
        elif r.y == 0:
            if r.x == 0:
                theta_obs[-1] = 1
            elif r.x > 0:
                theta_obs[0] = 1
            else:
                theta_obs[4] = 1
        else:
            if r.y > 0:
                theta_obs[2] = 1
            else:
                theta_obs[6] = 1

    return theta_obs


def get_edge_obs(patch_vec: Vector) -> np.array:
    edge_obs = np.array([patch_vec.y <= config.PATCH_VEC_LIM_MIN[0],
                         patch_vec.x <= config.PATCH_VEC_LIM_MIN[1],
                         config.PATCH_VEC_LIM_MAX[0] <= patch_vec.y,
                         config.PATCH_VEC_LIM_MAX[1] <= patch_vec.x
                         ])  # sign change when sitting on the border.
    return edge_obs


def calculate_u(plgs_enc, p_unenc_closest, patch_vec):
    u_enc, u_unenc, u = 0, 0, 0
    for p_enc in plgs_enc:
        l2 = calculate_l2_norm(p_enc.center_of_mass, patch_vec)
        u_enc += 1 / (1 + config.ALPHA_ENC * l2)
    if p_unenc_closest:
        l2 = calculate_l2_norm(p_unenc_closest.center_of_mass, patch_vec)
        u_unenc = 1 / (1 + config.ALPHA_UNENC * l2)

    u1_inv = config.NO_ENCLOSED_PENALTY if u_enc == 0 else 1 / u_enc
    u2_inv = 0 if u_unenc == 0 else 1 / u_unenc
    u = config.U_PREFACTOR / (u1_inv + u2_inv)
    return u


class StepAnimation(animation.TimedAnimation):
    def __init__(self, info, image_lr):
        self.info = info
        self.image_lr = image_lr
        self.total_steps = len(self.info)

        self._fig = plt.figure()
        self.ax = self._fig.add_subplot(1, 1, 1)
        # self.ax2 = self._fig.add_subplot(1, 2, 2)

        self.ax.axis('off')

    def _draw_frame(self, i):
        self.ax.clear()
        self.ax.imshow(self.image_lr)
        for rect in self.info[i]['rects']:
            self.ax.add_patch(rect)

        for line in self.info[i]['lines']:
            self.ax.add_line(line)

        out_str = 'turn: ' + str(i) + ', ' + \
            'u: ' + str(round(self.info[i]['u'], 3)) + ', ' + \
            'reward: ' + str(round(self.info[i]['reward'], 3)) + ', ' + \
            'cum_reward: ' + str(round(self.info[i]['cum_reward'], 3)) + ', ' + \
            'action: ' + config.ACTION_MAPPING_REV[self.info[i]['action']][0]

        plt.suptitle(out_str)

    def new_frame_seq(self):
        return iter(range(self.total_steps))


class EpisodeAnimation(animation.TimedAnimation):
    def __init__(self, info):
        self.info = info
        self.total_eps = len(self.info)
        self.total_plgs = self.info[0]['num_plgs']

        self._fig = plt.figure()
        self.ax = self._fig.add_subplot(1, 1, 1)

        self.ax.axis('off')

    def _draw_frame(self, i):
        self.ax.clear()
        self.ax.imshow(self.info[i]['image_lr'])
        self.ax.add_patch(self.info[i]['rects'][0])

        out_str = 'ep: ' + str(i) + '/' + str(self.total_eps) + ', ' + \
            'num_plgs: ' + str(self.info[i]['num_plgs']) + '/' + str(self.total_plgs) + ', ' + \
            'term_status: ' + str(config.TERMINATION_STATUS_REV[self.info[i]['term_status']])

        plt.suptitle(out_str)

    def new_frame_seq(self):
        return iter(range(self.total_eps))


class FinderAnimation(animation.TimedAnimation):
    def __init__(self, info, image_hr):
        self.image_hr = image_hr
        self.info = info
        self.total_eps = len(self.info)
        self.total_plgs = self.info[0]['num_plgs']

        self._fig = plt.figure(dpi=80)
        self.ax_hr = self._fig.add_subplot(1, 3, 1)
        self.ax_lr = self._fig.add_subplot(1, 3, 2)
        self.ax_crp = self._fig.add_subplot(1, 3, 3)
        # self.ax2 = self._fig.add_subplot(1, 2, 2)

        self.ax_hr.axis('off')
        detector_utils.plot_image_from_output(self.image_hr, annotations=None, ax=self.ax_hr)

    def _draw_frame(self, i):
        if self.ax_hr.patches:
            while self.ax_hr.patches:
                self.ax_hr.patches.pop()
        self.ax_hr.add_patch(self.info[i]['rect_hr'])

        self.ax_lr.clear()
        self.ax_lr.imshow(self.info[i]['image_lr'])
        self.ax_lr.add_patch(self.info[i]['rects_lr'][0])
        self.ax_lr.axis('off')

        self.ax_crp.clear()
        detector_utils.plot_image_from_output(self.info[i]['image_hr_crp'], annotations=self.info[i]['preds'], ax=self.ax_crp)
        self.ax_crp.axis('off')

        out_str = 'ep: ' + str(i) + '/' + str(self.total_eps) + ', ' + \
            'num_plgs: ' + str(self.info[i]['num_plgs']) + '/' + str(self.total_plgs) +', ' + \
            'num_xtals: ' + str(self.info[i]['num_xtals']) + ', ' + \
            'term_status: ' + str(config.TERMINATION_STATUS_REV[self.info[i]['term_status']])
        #
        plt.suptitle(out_str)
        self._fig.tight_layout()
        pass

    def new_frame_seq(self):
        return iter(range(self.total_eps))


def load_detector(location):
    detector = torchvision.models.detection.retinanet_resnet50_fpn(num_classes=len(detector_config.CRYSTAL_TYPE), pretrained=False, pretrained_backbone=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_path = config.DETECTOR_MODEL_PATH_LOCAL if location == 'local' else config.DETECTOR_MODEL_PATH_SERVER
    files = sorted([f for f in os.listdir(model_path) if f[-2:] == 'pt'])
    weight_path = os.path.join(model_path, files[-1])
    detector.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
    detector.to(device).eval()

    return detector, device

def load_rl_model_checkpoint(run_id):
    folder_name = None
    save_path_parent = 'rl_model/'
    folders = sorted(os.listdir(save_path_parent))
    for f in folders:
        if f[:3] == run_id:
            folder_name = f

    checkpoints_path = os.path.join('rl_model/', folder_name)
    checkpoints = [f for f in os.listdir(checkpoints_path) if f[-4:] == '.zip']
    max_len = 0
    for c in checkpoints:
        max_len = max(max_len, len(c))
    checkpoints = sorted([c for c in checkpoints if len(c) == max_len])
    model_checkpoint = checkpoints[-1]
    return checkpoints_path, model_checkpoint


def get_patch_coords_hr(patch_vec):
    ymin_hr, ymax_hr = (patch_vec.y - config.PATCH_LR_SHAPE[0] / 2) * config.RATIO_HR_LR, (patch_vec.y + config.PATCH_LR_SHAPE[0] / 2) * config.RATIO_HR_LR
    xmin_hr, xmax_hr = (patch_vec.x - config.PATCH_LR_SHAPE[1] / 2) * config.RATIO_HR_LR, (patch_vec.x + config.PATCH_LR_SHAPE[1] / 2) * config.RATIO_HR_LR

    return (int(ymin_hr), int(ymax_hr), int(xmin_hr), int(xmax_hr))


def check_spot(y, x, occ_pix, image_lr, directions):
    if image_lr[y][x] == 1:
        return False
    for (dy, dx) in directions:
        new_y = max(min(y+dy, config.IMAGE_LR_SHAPE[0]-1), 0)
        new_x = max(min(x+dx, config.IMAGE_LR_SHAPE[1]-1), 0)
        if image_lr[new_y][new_x] == 1 and (new_y, new_x) not in {n for n in occ_pix.values()}:
            return False
    return True


def make_rand_image_lr():
    num_plgs = np.random.normal(loc=74, scale=11.314)
    # num_plgs = np.random.uniform(low=0, high=100)
    num_plgs = num_plgs * -1 if num_plgs < 0 else num_plgs
    num_plgs = round(num_plgs)

    image_lr = np.zeros(config.IMAGE_LR_SHAPE)
    plgs = set()

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while num_plgs:
        start_over = False

        while 1:  # generate num pix
            num_pix = max(round(np.random.normal(loc=3.939, scale=3.502)), 1)
            if num_pix > config.NUM_PIXELS_MAX:
                continue
            else:
                break
        # plotting pixels
        p = Polygon()
        p.num_pixels = num_pix
        occ_pix = defaultdict(tuple)  # key for indexing, value for coordinates
        while num_pix:
            if not occ_pix:
                tries = 5
                while 1:
                    y = np.random.randint(0, config.IMAGE_LR_SHAPE[0]-1)
                    x = np.random.randint(0, config.IMAGE_LR_SHAPE[1]-1)
                    if check_spot(y, x, occ_pix, image_lr, directions):
                        break
                    tries -= 1
                    if tries == 0:
                        start_over = True
                        break
            else:
                tries = 5
                while 1:
                    idx = 0 if len(occ_pix) <= 1 else np.random.randint(0, len(occ_pix)-1)
                    dy, dx = directions[np.random.randint(0, len(directions)-1)]
                    y = occ_pix[idx][0] + dy
                    x = occ_pix[idx][1] + dx
                    y = max(min(y, config.IMAGE_LR_SHAPE[0]-1), 0)
                    x = max(min(x, config.IMAGE_LR_SHAPE[1]-1), 0)
                    if check_spot(y, x, occ_pix, image_lr, directions):
                        break
                    tries -= 1
                    if tries == 0:
                        start_over = True
                        break

            if start_over: # exit if stuck
                break
            else:
                image_lr[y][x] = 1
                occ_pix[p.num_pixels-num_pix] = (y, x)
                num_pix -= 1

        if start_over:
            for pix in occ_pix.values():
                image_lr[pix[0]][pix[1]] = 0
            continue
        else:
            p.loc = {Vector(v[0], v[1]) for v in occ_pix.values()}
            p.center_of_mass = Vector(sum([v[0] for v in occ_pix.values()]) / p.num_pixels,
                                      sum([v[1] for v in occ_pix.values()]) / p.num_pixels
                                      )
        plgs.add(p)
        num_plgs -= 1

    return image_lr, plgs
