import dtxtals.rl.core as core
import dtxtals.rl.config as config
import dtxtals.rl.utils as utils
import os
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from matplotlib import patches
from collections import defaultdict
from torchvision import transforms as T
import dtxtals.detector.detector_config as detector_config
import dtxtals.detector.detector_utils as detector_utils
import torch
import cv2
from datetime import datetime


class CrystalFinder(core.CrystalEnv):
    def __init__(self, folder_name, detector, device, **kwargs):
        super(CrystalFinder, self).__init__(folder_name, **kwargs)
        # Detector
        self.detector = detector
        self.device = device

        # High resolution images
        self.image_hr = None
        self.image_hr_crp = None

        # Keeping track of detected objects
        self.preds = None
        self.detected_objects = {crystal_type: {'boxes': torch.empty((0, 4)).to(self.device), 'scores': torch.empty(0,).to(self.device)}
                                 for crystal_type in detector_config.CRYSTAL_TYPE.keys()}
        self.new_good_xtal, self.new_bad_xtal = False, False

    def reset(self, legacy=None):
        super(CrystalFinder, self).reset(legacy=legacy)
        image_raw = cv2.imread(os.path.join(self.folder_name, self.image_name), cv2.IMREAD_COLOR)
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            self.image_hr = torch.as_tensor(image_raw, dtype=torch.float32).permute(2, 0, 1).to(self.device) / 255.0

        return self.state

    def search(self):
        coords = utils.get_patch_coords_hr(self.patch_vec)
        self.image_hr_crp = T.functional.crop(self.image_hr, top=coords[0], left=coords[2], height=config.PATCH_HR_SHAPE[0], width=config.PATCH_HR_SHAPE[1])
         # Get predictions on the patch using a detector.
        self.preds = detector_utils.predict_with_model(self.detector, self.image_hr_crp.unsqueeze(0), detector_config.PREDICTION_THRESHOLD)[0]  # dict

        # Global coordinates for bounding boxes in the format: x1y1x2y2
        with torch.no_grad():
            self.preds['boxes_global'] = torch.clone(self.preds['boxes']).detach()
            for idx, box in enumerate(self.preds['boxes_global']):
                self.preds['boxes_global'][idx] = torch.Tensor([coords[2]+box[0], coords[0]+box[1], coords[2]+box[2], coords[0]+box[3]])

        # Check if any new crystals were discovered. If so, append their bounding boxes.
        self.new_good_xtal, self.new_bad_xtal = False, False
        if all(self.preds['boxes_global'].shape):
            for idx, (box, score) in enumerate(zip(self.preds['boxes_global'], self.preds['scores'])):
                self.new_good_xtal, self.new_bad_xtal, add_this_xtal = False, False, True
                box = box.unsqueeze(0)
                for k in self.detected_objects.keys():
                    ious = detector_utils.get_iou(box, self.detected_objects[k]['boxes'])
                    if any(ious > config.IOU_NEW_CRYSTAL_THRESHOLD):
                        best_match_idx = torch.argmax(ious)
                        if detector_utils.get_area(box.squeeze(0)) > \
                           detector_utils.get_area(self.detected_objects[k]['boxes'][best_match_idx]):
                            self.detected_objects[k]['boxes'][best_match_idx] = box.squeeze(0)
                            self.detected_objects[k]['scores'][best_match_idx] = score
                        add_this_xtal = False
                        break
                if add_this_xtal:
                    object_type_str = detector_config.CRYSTAL_TYPE_REV[self.preds['labels'][idx].item()]
                    if object_type_str == 'good':
                        self.new_good_xtal = True
                    elif object_type_str == 'bad':
                        self.new_bad_xtal = True
                    self.detected_objects[object_type_str]['boxes'] = torch.cat([self.detected_objects[object_type_str]['boxes'], box])
                    self.detected_objects[object_type_str]['scores'] = torch.cat((self.detected_objects[object_type_str]['scores'], score.unsqueeze(0)))

    def render(self, show=True):
        rects_lr, lines = super(CrystalFinder, self).render(show=False)
        coords = utils.get_patch_coords_hr(self.patch_vec)
        rect_hr = patches.Rectangle(xy=(coords[2], coords[0]), width=coords[3] - coords[2], height=coords[1] - coords[0], linewidth=1, edgecolor='black', facecolor='none')

        if show:
            fig, axs = plt.subplots(ncols=3, figsize=(6, 3), dpi=80)
            axs[0].imshow(self.image_hr.cpu().permute(1, 2, 0))
            axs[0].axis('off')
            axs[0].add_patch(rect_hr)

            axs[1].imshow(self.image_lr)
            axs[1].axis('off')
            axs[1].add_patch(rects_lr[0])

            detector_utils.plot_image_from_output(self.image_hr_crp, self.preds, ax=axs[2])
            axs[2].axis('off')

            fig.tight_layout()
            plt.show()

        return rects_lr, rect_hr


def analyze_substrate(folder_name, image_idx, detector, device,
                      checkpoints_path, model_checkpoint, num_episodes=config.MAX_EPISODES, max_steps=config.MAX_STEPS,
                      save_replay=False, verbose=False, high_resolution=False, max_xtals=1,
                      filename=None, output_path=None):

    filename = 'file' if not filename else filename
    output_path = checkpoints_path if not output_path else output_path
    model = PPO.load(os.path.join(checkpoints_path, model_checkpoint))
    fndr = CrystalFinder(folder_name=folder_name,
                         image_idx=image_idx,
                         detector=detector,
                         device=device,
                         max_xtals=max_xtals
                         )
    fndr.seed()
    legacy = None
    new_start_location = False

    info = defaultdict(dict)
    run_info = {}

    for episode in range(num_episodes):
        if episode == 0:
            state = fndr.reset()
            num_plgs_0 = len(fndr.plgs)
        else:
            if new_start_location:
                patch_vec = utils.Vector(fndr.rng.randint(config.PATCH_VEC_LIM_MIN[0], config.PATCH_VEC_LIM_MAX[0]),
                                         fndr.rng.randint(config.PATCH_VEC_LIM_MIN[1], config.PATCH_VEC_LIM_MAX[1]),
                                         )
                legacy['patch_vec'] = patch_vec
                new_start_location = False
            state = fndr.reset(legacy=legacy)

        for step in range(max_steps):
            action = model.predict(state)
            state, reward, done, term_info = fndr.step(action[0])
            if done:
                break

        if term_info['term_status'] == config.TERMINATION_STATUS['out_of_bnd']:
            new_start_location = True
        else:
            if term_info['term_status'] == config.TERMINATION_STATUS['no_plg'] or \
               term_info['term_status'] == config.TERMINATION_STATUS['TO_fnd_plg'] or \
               term_info['term_status'] == config.TERMINATION_STATUS['TO_no_plg']:
                new_start_location = True
            for plg_enc in fndr.plgs_enc:
                for l in plg_enc.loc:
                    fndr.image_lr[l.y][l.x] = 0
                fndr.plgs.remove(plg_enc)

        last_ep_plgs = fndr.plgs.copy()
        last_ep_patch_vec = utils.Vector(fndr.patch_vec[0], fndr.patch_vec[1])
        legacy = {'plgs': last_ep_plgs,
                  'patch_vec': last_ep_patch_vec
                  }

        if term_info['term_status'] == config.TERMINATION_STATUS['fnd_plg']:
            fndr.search()  # Only zoom in when the agent is sure
        else:
            fndr.image_hr_crp = torch.zeros(3, config.PATCH_HR_SHAPE[0], config.PATCH_HR_SHAPE[1]).to(fndr.device)
            fndr.preds = {}

        rects_lr, rect_hr = fndr.render(show=False)
        num_xtals = len(fndr.detected_objects['good']['boxes'])
        if save_replay:
            info[episode] = {
                'image_lr': fndr.image_lr.copy(),
                'image_hr_crp': fndr.image_hr_crp.clone(),
                'rect_hr': rect_hr,
                'rects_lr': rects_lr,
                'num_plgs': len(fndr.plgs),
                'term_status': term_info['term_status'],
                'num_xtals': num_xtals,
                'preds': fndr.preds.copy()
            }

        if verbose:
            print('ep: {}, plgs: {}, xtals: {}, term_stat: {}'.format(episode,
                                                                  len(fndr.plgs),
                                                                  num_xtals,
                                                                  config.TERMINATION_STATUS_REV[term_info['term_status']]))
        if len(fndr.plgs) == 0:
            break

    run_info['ep'] = episode+1
    run_info['num_plgs_0'] = num_plgs_0
    run_info['num_xtals'] = num_xtals
    run_info['p_ep'] = round(num_plgs_0 / (episode+1), 3)

    if save_replay:
        print('Creating a GIF...')
        ani = utils.FinderAnimation(info=info, image_hr=fndr.image_hr)
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        ani.save(os.path.join(output_path, filename + '_fndr.gif'), fps=5, extra_args=[], dpi=100 if high_resolution else 80)
        print('GIF created.')
        print('done')

    return fndr.image_hr, fndr.detected_objects, run_info
