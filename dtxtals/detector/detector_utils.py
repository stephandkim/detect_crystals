import cv2
import os
import numpy as np
import json
import torch
import dtxtals.detector.detector_config as detector_config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset
import torchvision
import albumentations.pytorch
from tqdm import tqdm
plt.style.use('dark_background')


# Customized normalize_bbox method for fixing errors
def normalize_bbox(bbox, rows, cols):
    """Normalize coordinates of a bounding box. Divide x-coordinates by image width and y-coordinates
    by image height.
    """
    x_min, y_min, x_max, y_max = bbox[:4]
    normalized_bbox = [max(0, x_min / cols), max(0, y_min / rows), min(1, x_max / cols), min(1, y_max / rows)]
    return normalized_bbox + list(bbox[4:])

albumentations.core.bbox_utils.normalize_bbox = normalize_bbox


def get_target(annotations_path):
    # Load targets from the given path
    with open(annotations_path) as f:
        data = json.load(f)
        boxes = torch.as_tensor(data['boxes'], dtype=torch.float32)
        crystal_ids = torch.as_tensor(data['crystal_ids'], dtype=torch.int64)
        labels = []
        for label in data['labels']:
            labels.append(detector_config.CRYSTAL_TYPE[label])
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['crystal_ids'] = crystal_ids

        f.close()
    return target


def collate_fn(batch):
    return tuple(zip(*batch))


def plot_image_from_output(image: torch.Tensor, annotations, fig=None, ax=None, save=False, save_path=None, filename=None, is_target=False):
    # Plot the given tensor (image) with annotations.
    image = image.cpu().permute(1, 2, 0)
    if not ax:
        fig, ax = plt.subplots(1)
    ax.imshow(image, interpolation='none')
    ax.axis('off')

    if annotations:
        with torch.no_grad():
            annotations = {k: v.to('cpu') for k, v in annotations.items()}

        for idx in range(len(annotations['boxes'])):
            xmin, ymin, xmax, ymax = annotations['boxes'][idx]

            if annotations['labels'][idx] == 0:
                edgecolor = 'g' if is_target else 'r'
                rect = patches.Rectangle(xy=(xmin, ymin), width=(xmax - xmin), height=(ymax - ymin), linewidth=1,
                                         edgecolor=edgecolor, facecolor='none')

            ax.add_patch(rect)
    if not ax:
        plt.show()
    if save:
        fig.savefig(os.path.join(save_path, filename + '.png'), dpi=300)


class CrystalDataset(Dataset):
    # Dataset object for crystal image data.

    def __init__(self, path, transform=None, image_idx=None):
        self.path = path
        self.images = sorted(os.listdir(os.path.join(self.path, 'images')))
        if image_idx:
            self.images = [self.images[n] for n in range(len(self.images)) if n in image_idx]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_image = self.images[idx]
        file_annotations = file_image[:-3] + 'json'
        image_path = os.path.join(self.path, 'images', file_image)
        annotations_path = os.path.join(self.path, 'annotations', file_annotations)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        if not self.transform:
            to_tensor = torchvision.transforms.ToTensor()
            image = to_tensor(image)
            target = get_target(annotations_path)
        else:
            target = get_target(annotations_path)
            len_bbox = 0
            with torch.no_grad():
                while len_bbox == 0:
                    # In case of nontrivial transformation, repeat the transformation until there is at least
                    # one target in the image.
                    transformed = self.transform(image=image, bboxes=target['boxes'], labels=target['labels'])
                    len_bbox = len(transformed['bboxes'])
                image = transformed['image']
                target = {'boxes': torch.as_tensor(transformed['bboxes'], dtype=torch.float32),
                          'labels': torch.as_tensor(transformed['labels'], dtype=torch.int64)}

        return image, target


def predict_with_model(model, images, threshold):
    # Predict the given images with the model. Return predictions with scores above the threshold.
    model.eval()
    with torch.no_grad():
        preds = model(images)
    for image_id in range(len(preds)):
        idx_list = []

        for idx, score in enumerate(preds[image_id]['scores']):
            if score > threshold:
                idx_list.append(idx)

        preds[image_id]['boxes'] = preds[image_id]['boxes'][idx_list]
        preds[image_id]['labels'] = preds[image_id]['labels'][idx_list]
        preds[image_id]['scores'] = preds[image_id]['scores'][idx_list]

    return preds


def get_batch_predictions(model_path, test_dataset_loader, anchor_generator=None):
    # Return predictions for a batch of images from a PyTorch dataset loader object.
    files = sorted([f for f in os.listdir(model_path) if f[-2:] == 'pt'])
    weight_path = os.path.join(model_path, files[-1])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    retina = torchvision.models.detection.retinanet_resnet50_fpn(num_classes=len(detector_config.CRYSTAL_TYPE),
                                                                 pretrained=False,
                                                                 anchor_generator=anchor_generator,
                                                                 pretrained_backbone=True
                                                                 )
    retina.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
    retina.to(device).eval()

    with torch.no_grad():
        labels = torch.Tensor([])
    annots = []
    preds = []

    for images, targets in tqdm(test_dataset_loader, position=0, leave=True):
        images = [image.to(device) for image in images]

        labels = torch.cat([labels, torch.cat([t['labels'] for t in targets])])

        with torch.no_grad():
            preds_batch = predict_with_model(retina, images, threshold=0.5)
            preds_batch = [{k: v.to(torch.device('cpu')) for k, v in p.items()} for p in preds_batch]
            preds.append(preds_batch)
            annots.append(targets)

    return preds, annots, labels


def get_iou(box1, box2):
    # Calculate the intersection over union of the two given (sets of) tensors.
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    int_x1 = torch.max(box1_x1, box2_x1)
    int_y1 = torch.max(box1_y1, box2_y1)
    int_x2 = torch.min(box1_x2, box2_x2)
    int_y2 = torch.min(box1_y2, box2_y2)

    intersection = torch.clamp(int_x2 - int_x1, min=0) * torch.clamp(int_y2 - int_y1, min=0)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = box1_area + box2_area - intersection
    iou = torch.clamp(intersection / union, min=0, max=1)

    return iou


def get_area(box):  #x1y1x2y2
    return abs(box[0] - box[2]) * abs(box[1] - box[3])


def get_positives(pred, annot, iou_threshold):
    # Compare predictions and annotations, and get true positives.
    res = []
    for substrate_id in range(len(pred)):
        boxes_pred = pred[substrate_id]['boxes']
        boxes_annot = annot[substrate_id]['boxes']
        visited = set()

        with torch.no_grad():
            positives = torch.zeros(len(boxes_pred))

        for crystal_id, box_pred in enumerate(boxes_pred):
            max_iou, match_idx = get_iou(box_pred.unsqueeze(0), boxes_annot).max(0)
            if max_iou > iou_threshold and match_idx not in visited:
                positives[crystal_id] = 1
                visited.add(match_idx)

        res.append([positives, pred[substrate_id]['scores'], pred[substrate_id]['labels']])

    return res


def get_average_precision(precision_curve, recall_curve):
    precision_curve = np.array([0.0, *precision_curve, 0.0])
    recall_curve = np.array([0.0, *recall_curve, 1.0])

    for idx in range(len(recall_curve)-1, 0, -1):
        precision_curve[idx-1] = max(precision_curve[idx], precision_curve[idx-1])

    idx_list = [0]
    for idx in range(1, len(recall_curve)):
        if recall_curve[idx] != recall_curve[idx-1]:
            idx_list.append(idx)
    idx_list = np.array(idx_list)

    average_precision = np.sum((recall_curve[idx_list[1:]] - recall_curve[idx_list[:-1]]) * precision_curve[idx_list[1:]])

    return average_precision


def calculate_pc_metrics(preds, annots, labels):
    res = []
    for batch_id in range(len(preds)):
        res.append(get_positives(preds[batch_id], annots[batch_id], iou_threshold=detector_config.IOU_THRESHOLD))

    pos, scores_pred, class_pred = [torch.cat([torch.cat([substrate[x] for substrate in batch]) for batch in res]) for x in range(3)]
    sorted_idx_list = torch.sort(-scores_pred)[1]

    pos = pos[sorted_idx_list]
    scores_preds = scores_pred[sorted_idx_list]
    class_pred = class_pred[sorted_idx_list]

    class_unique = torch.unique(class_pred)

    precision, recall, f1, average_precision = [], [], [], []
    for c in class_unique:
        true_pos_idx = class_pred == c
        num_ground_truth = (labels == c).sum()

        if num_ground_truth == 0 or true_pos_idx.sum() == 0:
            precision.append(0)
            recall.append(0)
            f1.append(0)

        else:
            true_pos_cumsum = torch.cumsum(pos[true_pos_idx], dim=-1)
            false_pos_cumsum = torch.cumsum(1-pos[true_pos_idx], dim=-1)

            precision_curve = true_pos_cumsum/(true_pos_cumsum + false_pos_cumsum)
            recall_curve = true_pos_cumsum/num_ground_truth

            precision.append(precision_curve[-1])
            recall.append(recall_curve[-1])
            f1.append(2 / (1/precision_curve[-1] + 1/recall_curve[-1]))
            average_precision.append(get_average_precision(precision_curve, recall_curve))

    return precision, recall, f1, average_precision, [detector_config.CRYSTAL_TYPE_REV[n.item()] for n in class_unique]
