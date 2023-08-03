import numpy as np
import torch

from config.config import Configs
from pycocotools.coco import COCO
import json
import cv2
from ssd.utils_ssd300 import dboxes300_coco
from train_model.train_loader import  CocoDataReader
from ssd.utils_ssd300 import calc_iou_tensor


def get_coco_ground_truth():
    val_annotate = "train_model/instances_val2017.json"
    cocoGt = COCO(annotation_file=val_annotate)
    return cocoGt

def main_train():
    configs = Configs()
    torch.manual_seed(configs.random_seed)
    np.random.seed(seed=configs.random_seed)
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
    img, img_id, size_image, bbox_sizes, bbox_labels = datareader[0]
    dboxes(order="ltrb")
    #ios = calc_iou_tensor()









if __name__ == "__main__":
    main_train()
