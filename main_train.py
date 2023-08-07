
import numpy as np
import torch
from PIL import ImageDraw
from pycocotools.coco import COCO

from config.config import Configs
from ssd.utils_ssd300 import dboxes300_coco
from train_model.train_loader import CocoDataReader
from utils.utils import SSDCropping


def get_coco_ground_truth():
    val_annotate = "train_model/instances_val2017.json"
    cocoGt = COCO(annotation_file=val_annotate)
    return cocoGt

def main_train():
    configs = Configs()
    torch.manual_seed(0)
    np.random.seed(seed=0)
    #cocoGt =  get_coco_ground_truth()
    #print(cocoGt) # coco аннотации, пока решил сделать все как в исходнике

    dboxes = dboxes300_coco()
    #print(dboxes.dboxes.shape) # создал дефолтные bbx [8732, 4]
    annotate_file = 'COCOdata\\annotations\\instances_val2017.json'
    datareader = CocoDataReader(
        img_folder='COCOdata\\val2017\\',
        annotate_file=annotate_file)

    #print(datareader.img_keys)

     #input ltrb format, output xywh format
    img, img_id, size_image, bbox_sizes, bbox_labels = datareader[404484]

    dboxes(order="ltrb")
    #ios = calc_iou_tensor()
    #flip = RandomHorizontalFlip()
    crop = SSDCropping()
    new_img, _, bboxes, labels = crop(

        img, size_image, bbox_sizes, bbox_labels)
    #img.show()
    #new_img.show()
    draw = ImageDraw.Draw(new_img)
    htot, wtot = size_image
    for i in bboxes:
        l,t,r,b = i.tolist()

        draw.rectangle(((int(l * wtot), int(t * htot)), (int(r * wtot), int(b * htot))))
    new_img.show()
        #b = t + h
        #l, t, r, b = xc - 0.5*w, yc - 0.5*h, xc + 0.5*w, yc + 0.5*h
       # bbox_size = (l/wtot, t/htot, r/wtot, b/htot)
        #bbox_sizes.append(bbox_size)
       # bbox_labels.append(bbox_label)

       # bbox_sizes = torch.tensor(bbox_sizes)
       # bbox_labels =  torch.tensor(bbox_labels)










if __name__ == "__main__":
    main_train()
