import sys
from pathlib import Path
import cv2
import torch
import torch.nn as nn
from torchsummary import summary
import os


from ssd.model import ResNet
from ssd.entrypoints import _download_checkpoint, nvidia_ssd
from ssd.nvidia_ssd_processing_utils import Processing as processing


def main():
    return 0
 
        


if __name__ == "__main__":
    IMAGE_SIZE = (300, 300)
    backbone = ResNet()
    model = nvidia_ssd() 
    model.eval()
    uris = [
    'http://images.cocodataset.org/val2017/000000397133.jpg',
    'http://images.cocodataset.org/val2017/000000037777.jpg',
    'http://images.cocodataset.org/val2017/000000252219.jpg'
     ]
    inputs = [processing.prepare_input(uri) for uri in uris]
    #precision = 'fp32'
    tensor = processing.prepare_tensor(inputs)
    with torch.no_grad():
        detections_batch = model(tensor)
    results_per_input = processing.decode_results(detections_batch)
    best_results_per_input = [processing.pick_best(results, 0.40) for results in results_per_input]
    
    bboxes, classes, confidences = best_results_per_input[2]
    
    image = inputs[2].copy()
    orig_h, orig_w = image.shape[0], image.shape[1]
    print(classes, bboxes)
    for idx in range(len(bboxes)):
        if classes[idx] == 1:
            # get the bounding box coordinates in xyxy format
            x1, y1, x2, y2 = bboxes[idx]
            # resize the bounding boxes from the normalized to 300 pixels
            x1, y1 = int(x1*300), int(y1*300)
            x2, y2 = int(x2*300), int(y2*300)
            # resizing again to match the original dimensions of the image
            x1, y1 = int((x1/300)*orig_w), int((y1/300)*orig_h)
            x2, y2 = int((x2/300)*orig_w), int((y2/300)*orig_h)
            # draw the bounding boxes around the objects
            cv2.rectangle(
                image, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA
            )
    
    
    
    cv2.imshow('image', image)
    cv2.imwrite()
    cv2.waitKey(0)
    
    
    #main()
