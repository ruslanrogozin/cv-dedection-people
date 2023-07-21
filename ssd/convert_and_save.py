import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
from ssd.nvidia_ssd_processing_utils import Processing as processing
import cv2

def convert_and_save(prediction, original_image, name, format_data,  threshold=0.5, path='new_data/'):
    ''' function for convetr and save pictures'''
    original = cv2.imread(str(original_image)) 

    best_results_per_input = processing.pick_best(prediction[0], threshold)
    bboxes, classes, confidences = best_results_per_input

    
    orig_h, orig_w = original.shape[0], original.shape[1]

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
                original, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA
            )
            
    #cv2.imshow('image', original)
    #cv2.waitKey(0)
    cv2.imwrite(path + 'new_' + name + '.' + format_data, original)
