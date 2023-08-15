import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

from config.config import Configs


def set_seed(seed=Configs.random_seed):
    """Делает наши результаты воспроизводимыми (вычисления могут немного больше времени занимать)"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")


def draw_bboxes(
    prediction,
    original,
    use_padding=Configs.use_padding_in_image_transform,
    use_head=Configs.use_head,
):
    if isinstance(original, str):
        original = cv2.imread(original)

    elif isinstance(original, np.ndarray):
        original = original.copy()

    else:
        original = cv2.imread(str(original))

    original_shape = (original.shape[0], original.shape[1])

    bboxes, classes, _ = prediction

    if use_padding:
        orig_h = max(original.shape[0], original.shape[1])
        orig_w = orig_h
        # calculate shapes for padding
        new_shape = (300, 300)
        ratio = float(max(new_shape)) / max(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])
        # original  = cv2.resize(original, new_size)
        new_h, new_w = new_size
        delta_h = 300 - new_h
        delta_w = 300 - new_w

    else:
        orig_h, orig_w = original.shape[0], original.shape[1]
        delta_h = 0
        delta_w = 0

    info = {1: [(0, 0, 255), "person"], 2: [(0, 255, 0), "head"]}

    for idx, bbox in enumerate(bboxes):
        if not use_head and (classes[idx] != 1):
            continue
        x1, y1, x2, y2 = bbox
        # resize the bounding boxes from the normalized to 300 pixels
        x1, y1 = int(x1 * 300) - delta_w // 2, int(y1 * 300) - delta_h // 2
        x2, y2 = int(x2 * 300) - delta_w // 2, int(y2 * 300) - delta_h // 2
        # resizing again to match the original dimensions of the image
        x1, y1 = int((x1 / 300) * orig_w), int((y1 / 300) * orig_h)
        x2, y2 = int((x2 / 300) * orig_w), int((y2 / 300) * orig_h)
        # draw the bounding boxes around the objects
        cv2.rectangle(
            original, (x1, y1), (x2, y2), info[classes[idx]][0], 2, cv2.LINE_AA
        )

        cv2.putText(
            original,
            info[classes[idx]][1],
            (x1, y1 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            info[classes[idx]][0],
            1,
        )

    return original


def save_model(
    model, optimizer=None, model_name="model_name", path="weight", lr_scheduler=None
):
    if isinstance(path, str):
        path = Path(path)
    path_save_state = path / ("state " + model_name + ".pth")
    state = {
        "model_name": model_name,
        "model": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "lr_scheduler_state": None
        if lr_scheduler is None
        else lr_scheduler.state_dict(),
    }
    torch.save(state, path_save_state)
