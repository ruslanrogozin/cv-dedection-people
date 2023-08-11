from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")


def draw_bboxes(prediction, original, use_padding=True):
    if isinstance(original, str):
        original = cv2.imread(original)

    elif isinstance(original, np.ndarray):
        original = original.copy()

    else:
        original = cv2.imread(str(original))

    original_shape = (original.shape[0], original.shape[1])

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

    bboxes, classes, _ = prediction

    for idx, bbox in enumerate(bboxes):
        if classes[idx] == 1:
            # get the bounding box coordinates in xyxy format
            x1, y1, x2, y2 = bbox
            # resize the bounding boxes from the normalized to 300 pixels
            x1, y1 = int(x1 * 300) - delta_w // 2, int(y1 * 300) - delta_h // 2
            x2, y2 = int(x2 * 300) - delta_w // 2, int(y2 * 300) - delta_h // 2
            # resizing again to match the original dimensions of the image
            x1, y1 = int((x1 / 300) * orig_w), int((y1 / 300) * orig_h)
            x2, y2 = int((x2 / 300) * orig_w), int((y2 / 300) * orig_h)
            # draw the bounding boxes around the objects
            cv2.rectangle(original, (x1, y1), (x2, y2),
                          (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(
                original,
                "person",
                (x1, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 25, 255),
                2,
            )

    return original


def save_model(
    model, optimizer=None, model_name="model_name", path="weight",
    lr_scheduler=None
):
    if isinstance(path, str):
        path = Path(path)
    path_save_state = path / ("state " + model_name + ".pth")
    state = {
        "model_name": model_name,
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "lr_scheduler_state": None
        if lr_scheduler is None
        else lr_scheduler.state_dict(),
    }
    torch.save(state, path_save_state)
