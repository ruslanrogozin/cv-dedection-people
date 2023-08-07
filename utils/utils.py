import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

from ssd.utils_ssd300 import calc_iou_tensor


class SquarePad(object):
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, bboxes):
        if random.random() < self.p:
            bboxes[:, 0], bboxes[:, 2] = 1.0 - bboxes[:, 2], 1.0 - bboxes[:, 0]
            return image.transpose(Image.FLIP_LEFT_RIGHT), bboxes
        return image, bboxes


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

            cv2.putText(original, "person", (x1, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 25, 255), 2)

    return original


class SSDCropping(object):

    """ Cropping for SSD, according to original paper
        Choose between following 3 conditions:
        1. Preserve the original image
        2. Random crop minimum IoU is among 0.1, 0.3, 0.5, 0.7, 0.9
        3. Random crop
        Reference to https://github.com/chauhan-utk/src.DomainAdaptation
    """

    def __init__(self):

        self.sample_options = (
            # Do nothing
            None,
            # min IoU, max IoU
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # no IoU requirements
            (None, None),
        )

    def __call__(self, img, img_size, bboxes, labels):

        # Ensure always return cropped image
        while True:
            mode = random.choice(self.sample_options)
            print(mode)
            if mode is None:
                return img, img_size, bboxes, labels

            htot, wtot = img_size

            min_iou, max_iou = mode
            min_iou = float("-inf") if min_iou is None else min_iou
            max_iou = float("+inf") if max_iou is None else max_iou

            # Implementation use 50 iteration to find possible candidate
            for _ in range(1):
                # print('nen')

                # suze of each sampled path in [0.1, 1] 0.3*0.3 approx. 0.1
                w = random.uniform(0.3, 1.0)
                h = random.uniform(0.3, 1.0)

                if w/h < 0.5 or w/h > 2:
                    continue

                # left 0 ~ wtot - w, top 0 ~ htot - h
                left = random.uniform(0, 1.0 - w)
                top = random.uniform(0, 1.0 - h)

                right = left + w
                bottom = top + h

                ious = calc_iou_tensor(bboxes, torch.tensor(
                    [[left, top, right, bottom]]))

                # tailor all the bboxes and return
                if not ((ious > min_iou) & (ious < max_iou)).all():
                    continue

                # discard any bboxes whose center not in the cropped image
                xc = 0.5*(bboxes[:, 0] + bboxes[:, 2])
                yc = 0.5*(bboxes[:, 1] + bboxes[:, 3])

                masks = (xc > left) & (xc < right) & (yc > top) & (yc < bottom)

                # if no such boxes, continue searching again
                if not masks.any():
                    continue

                bboxes[bboxes[:, 0] < left, 0] = left
                bboxes[bboxes[:, 1] < top, 1] = top
                bboxes[bboxes[:, 2] > right, 2] = right
                bboxes[bboxes[:, 3] > bottom, 3] = bottom

                bboxes = bboxes[masks, :]
                labels = labels[masks]

                left_idx = int(left*wtot)
                top_idx = int(top*htot)
                right_idx = int(right*wtot)
                bottom_idx = int(bottom*htot)
                img = img.crop((left_idx, top_idx, right_idx, bottom_idx))

                bboxes[:, 0] = (bboxes[:, 0] - left)/w
                bboxes[:, 1] = (bboxes[:, 1] - top)/h
                bboxes[:, 2] = (bboxes[:, 2] - left)/w
                bboxes[:, 3] = (bboxes[:, 3] - top)/h

                htot = bottom_idx - top_idx
                wtot = right_idx - left_idx
                img.show()
                return img, (htot, wtot), bboxes, labels


# Do data augumentation
class SSDTransformer(object):
    """ SSD Data Augumentation, according to original paper
        Composed by several steps:
        Cropping
        Resize
        Flipping
        Jittering
    """

    def __init__(self, dboxes, size=(300, 300), val=False):

        # define vgg16 mean
        self.size = size
        self.val = val

        self.dboxes_ = dboxes  # DefaultBoxes300()
        self.encoder = Encoder(self.dboxes_)

        self.crop = SSDCropping()
        self.img_trans = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ColorJitter(brightness=0.125, contrast=0.5,
                                   saturation=0.5, hue=0.05
                                   ),
            transforms.ToTensor()
        ])
        self.hflip = RandomHorizontalFlip()

        # All Pytorch Tensor will be normalized
        # https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.trans_val = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            # ToTensor(),
            self.normalize,])

    @property
    def dboxes(self):
        return self.dboxes_

    def __call__(self, img, img_size, bbox=None, label=None, max_num=200):
        # img = torch.tensor(img)
        if self.val:
            bbox_out = torch.zeros(max_num, 4)
            label_out = torch.zeros(max_num, dtype=torch.long)
            bbox_out[:bbox.size(0), :] = bbox
            label_out[:label.size(0)] = label
            return self.trans_val(img), img_size, bbox_out, label_out

        img, img_size, bbox, label = self.crop(img, img_size, bbox, label)
        img, bbox = self.hflip(img, bbox)

        img = self.img_trans(img).contiguous()
        img = self.normalize(img)

        bbox, label = self.encoder.encode(bbox, label)

        return img, img_size, bbox, label
