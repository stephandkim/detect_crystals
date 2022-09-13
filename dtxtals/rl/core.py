import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import cv2
import os
from collections import deque
import dtxtals.rl.utils as utils
import dtxtals.rl.config as config
from matplotlib import lines, patches
import matplotlib.pyplot as plt


class CrystalEnv(gym.Env):
    def __init__(self, folder_name, image_idx=None, random_start=True, start_vec=None, max_xtals=1):
        # Image variables
        self.folder_name = folder_name
        self.image_idx = image_idx
        self.image_name = None
        self.image_raw = None
        self.image_lr = None
        self.plgs = None
        self.plgs_enc, self.plgs_unenc = None, None  # enclosed and unenclosed polygons.
        self.p_unenc_closest = None  # closest unenclosed polygon.
        self.patch_vec = None # patch vector; center of mass of the zoom-in patch.
        self.max_xtals = max_xtals

        # Starting conditions
        self.random_start = random_start
        self.start_vec = start_vec

        # Observation and action spaces
        self.observation_space = spaces.Box(low=0, high=1, shape=(22, ), dtype=np.float32)
        self.action_space = spaces.Discrete(9)
        self.reward_range = (-1, 1)

        # Observations
        self.theta_obs_enc, self.theta_obs_unenc = None, None  # theta observations for enclosed and unenclosed polygons.
        self.edge_obs = None  # edge observations.
        self.num_enc_plg = None

        # State
        self.state = None

        # Potential scores
        self.u = None

        # Variables for termination
        self.done = None
        self.reward_tracker = deque([])
        self.counter = None
        self.term_status = None
        self.out_of_bound_flag = None

        # Seeding
        self.seed_val = None
        self.rng, seed = None, None

    def seed(self, seed_val=None):
        self.seed_val = seed_val
        self.rng, seed = seeding.np_random(self.seed_val)
        return [seed]

    def reset(self, legacy=None):
        if not legacy:
            if self.random_start:
                self.patch_vec = utils.Vector(self.rng.randint(config.PATCH_VEC_LIM_MIN[0], config.PATCH_VEC_LIM_MAX[0]),
                                              self.rng.randint(config.PATCH_VEC_LIM_MIN[1], config.PATCH_VEC_LIM_MAX[1])
                                              )
            else:
                self.patch_vec = self.start_vec

            if self.folder_name == config.FOLDER_NAMES['rand_image_lr']:
                self.image_lr, self.plgs = utils.make_rand_image_lr()
            else:
                # Load a high resolution image.
                image_names = sorted(os.listdir(self.folder_name))
                if self.image_idx:
                    image_names = [image_names[n] for n in self.image_idx]
                self.image_name = image_names[self.rng.randint(0, len(image_names))]
                image_raw = cv2.imread(os.path.join(self.folder_name, self.image_name), cv2.IMREAD_COLOR)
                image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

                # Perform the kmeans clustering on the high resolution image.
                self.image_lr = utils.create_image_lr(image_raw)

                # Based on the clustering, create a low resolution image.
                self.plgs = utils.count_polygons(self.image_lr)

        else:
            self.plgs, self.patch_vec = legacy['plgs'], legacy['patch_vec']

        # Count number of polygons in the patch.
        self.plgs_enc, self.plgs_unenc = utils.sort_polygons(self.patch_vec, self.plgs)

        # Find the closest unenclosed polygon.
        self.p_unenc_closest = utils.get_closest_unenclosed_polygons(self.plgs_unenc, self.patch_vec)

        # Update observations and calculate the potential score.
        self.theta_obs_enc = utils.get_theta_obs(plgs=self.plgs_enc, patch_vec=self.patch_vec, plgs_type='enclosed')
        self.theta_obs_unenc = utils.get_theta_obs(plgs=deque([self.p_unenc_closest]), patch_vec=self.patch_vec, plgs_type='unenclosed')
        self.edge_obs = utils.get_edge_obs(self.patch_vec)

        self.num_enc_plg = np.array([0]) if not self.plgs_enc else np.array([min(len(self.plgs_enc) / self.max_xtals, 1)])
        self.u = utils.calculate_u(self.plgs_enc, self.p_unenc_closest, self.patch_vec)

        self.state = np.concatenate([self.theta_obs_enc, self.theta_obs_unenc, self.edge_obs, self.num_enc_plg])

        self.done = False
        self.counter = 0
        self.term_status = None
        self.out_of_bound_flag = False

        return self.state

    def step(self, action):
        (dy, dx) = config.ACTION_MAPPING_REV[action][-1]
        new = [0, 0]
        for idx, (now, inc) in enumerate(zip(self.patch_vec, (dy, dx))):
            new[idx] = min(max(now + inc, config.PATCH_VEC_LIM_MIN[idx]), config.PATCH_VEC_LIM_MAX[idx])
            if now + inc < config.PATCH_VEC_LIM_MIN[idx] or config.PATCH_VEC_LIM_MAX[idx] < now + inc:
                self.out_of_bound_flag = True
        self.patch_vec = utils.Vector(new[0], new[1])

        self.plgs_enc, self.plgs_unenc = utils.sort_polygons(self.patch_vec, self.plgs)
        self.p_unenc_closest = utils.get_closest_unenclosed_polygons(self.plgs_unenc, self.patch_vec)

        self.theta_obs_enc = utils.get_theta_obs(plgs=self.plgs_enc, patch_vec=self.patch_vec, plgs_type='enclosed')
        self.theta_obs_unenc = utils.get_theta_obs(plgs=deque([self.p_unenc_closest]), patch_vec=self.patch_vec, plgs_type='unenclosed')
        self.edge_obs = utils.get_edge_obs(self.patch_vec)
        self.num_enc_plg = np.array([0]) if not self.plgs_enc else np.array([min(len(self.plgs_enc) / self.max_xtals, 1)])
        new_u = utils.calculate_u(self.plgs_enc, self.p_unenc_closest, self.patch_vec)

        self.state = np.concatenate([self.theta_obs_enc, self.theta_obs_unenc, self.edge_obs, self.num_enc_plg])

        discount_factor = 1 - self.counter / config.MAX_STEPS
        reward = discount_factor * config.REWARD['step'] if new_u > self.u else -1 * config.REWARD['step']
        self.u = new_u

        if self.reward_tracker and len(self.reward_tracker) > config.GRACE_PERIOD:
            self.reward_tracker.popleft()
        self.reward_tracker.append(reward)

        if config.ACTION_MAPPING_REV[action][0] == 'stop':  # agent voluntarily stopped.
            self.done = True
            if self.plgs_enc:
                reward = config.REWARD['fnd_plg'] * self.num_enc_plg[0]
                self.term_status = config.TERMINATION_STATUS['fnd_plg']
            else:
                reward = config.REWARD['no_plg']
                self.term_status = config.TERMINATION_STATUS['no_plg']
        elif self.out_of_bound_flag:
            self.done = True
            reward = config.REWARD['out_of_bnd']
            self.term_status = config.TERMINATION_STATUS['out_of_bnd']
        elif self.counter >= config.MAX_STEPS - 1:
            self.done = True
            if self.plgs_enc:
                reward = config.REWARD['TO_fnd_plg']
                self.term_status = config.TERMINATION_STATUS['TO_fnd_plg']
            else:
                reward = config.REWARD['TO_no_plg']
                self.term_status = config.TERMINATION_STATUS['TO_no_plg']
        self.counter += 1

        return self.state, reward, self.done, {'term_status': self.term_status}

    def render(self, show=True, title=None):
        ymin, ymax = self.patch_vec.y - config.PATCH_LR_SHAPE[0]/2, self.patch_vec.y + config.PATCH_LR_SHAPE[0]/2
        xmin, xmax = self.patch_vec.x - config.PATCH_LR_SHAPE[1]/2, self.patch_vec.x + config.PATCH_LR_SHAPE[1]/2
        ymin_outer, ymax_outer = ymin - 10, ymax + 10
        xmin_outer, xmax_outer = xmin - 10, xmax + 10
        xmid, ymid = abs(xmax - xmin) / 2 + xmin, abs(ymax - ymin) / 2 + ymin
        xdiff, ydiff = abs((xmid - xmin_outer)) / 2, abs((ymid - ymin_outer)) / 2

        rect = patches.Rectangle(xy=(xmin, ymin), width=(xmax - xmin), height=(ymax - ymin), linewidth=1, edgecolor='w', facecolor='none')
        rect_outer = patches.Rectangle(xy=(xmin_outer, ymin_outer), width=(xmax_outer-xmin_outer), height=(ymax_outer-ymin_outer), linewidth=1, edgecolor='w', facecolor='none')
        line1 = lines.Line2D([xmin_outer+xdiff, xmax_outer-xdiff], [ymin_outer, ymax_outer])
        line2 = lines.Line2D([xmax_outer-xdiff, xmin_outer+xdiff], [ymin_outer, ymax_outer])
        line3 = lines.Line2D([xmin_outer, xmax_outer], [ymin_outer+ydiff, ymax_outer-ydiff])
        line4 = lines.Line2D([xmin_outer, xmax_outer], [ymax_outer-ydiff, ymin_outer+ydiff])

        if show:
            fig, ax = plt.subplots(1)
            ax.imshow(self.image_lr)
            ax.add_patch(rect)
            if title:
                ax.set_title(title)
            # ax.add_patch(rect_outer)
            # ax.add_line(line1)
            # ax.add_line(line2)
            # ax.add_line(line3)
            # ax.add_line(line4)
            plt.show()

        else:
            return [rect, rect_outer], [line1, line2, line3, line4]
