import os
import dtxtals.rl.config as config
import dtxtals.rl.utils as utils
import dtxtals.rl.finder as finder
import torch
import torchvision
import dtxtals.detector.detector_config as detector_config
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser(description='Dynamic zoom-in detection for exfoliated crystals.')
    parser.add_argument('--rl_run_id', help='RL model id', type=str, default='NP3')
    parser.add_argument('--image_path', help='Path for evaluation images', type=str, default=config.FOLDER_NAMES['images'])
    parser.add_argument('--target_path', help='Path for evaluation targets', type=str, default=config.FOLDER_NAMES['annotations'])
    args = parser.parse_args()

    return args


def find_xtals(args):
    rl_run_id = args.rl_run_id

    # Load RL agent
    idx_list = config.FINITE_PLGS
    rl_checkpoints_path, rl_model_checkpoint = utils.load_rl_model_checkpoint(rl_run_id)
    eval_image_path = args.image_path
    eval_image_names = sorted(os.listdir(eval_image_path))

    # # Targets for calculating the AP metrics
    # eval_target_path = args.target_path
    # eval_target_names = sorted(os.listdir(eval_target_path))

    # Load detector
    detector = torchvision.models.detection.retinanet_resnet50_fpn(num_classes=len(detector_config.CRYSTAL_TYPE),
                                                                   pretrained=False, pretrained_backbone=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    files = sorted([f for f in os.listdir(config.DETECTOR_WEIGHT_PATH) if f[-2:] == 'pt'])
    detector.load_state_dict(torch.load(os.path.join(config.DETECTOR_WEIGHT_PATH, files[-1]), map_location=torch.device(device)))
    detector.to(device).eval()

    for idx in idx_list:
        print('Finding crystals on {}'.format(eval_image_names[idx]))
        filename = args.rl_run_id + '_' + eval_image_names[idx][:-4]
        output_path = 'output/'

        _, preds, _ = finder.analyze_substrate(folder_name=eval_image_path,
                                               image_idx=[idx],
                                               detector=detector,
                                               device=device,
                                               checkpoints_path=rl_checkpoints_path,
                                               model_checkpoint=rl_model_checkpoint,
                                               verbose=True,
                                               save_replay=True,
                                               output_path=output_path,
                                               filename=filename
                                               )

        # # For calculating AP metrics
        # targets = detector_utils.get_target(os.path.join(eval_target_path, eval_target_names[idx]))
        # for k in preds.keys():
        #     with torch.no_grad():
        #         preds[k]['labels'] = torch.zeros(preds[k]['scores'].shape).to(torch.device('cpu'))
        #     for j in preds[k].keys():
        #         preds[k][j] = preds[k][j].to(torch.device('cpu'))

        print('{} done'.format(eval_image_names[idx]))
        print('========================')

if __name__ == '__main__':
    args = get_args()
    find_xtals(args)
