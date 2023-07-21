import sys
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torchsummary import summary
import os
import cv2


from ssd.model import ResNet
from ssd.convert_and_save import convert_and_save
from ssd.entrypoints import _download_checkpoint, nvidia_ssd
from ssd.nvidia_ssd_processing_utils import Processing as processing
from  ssd.dataloader import ImagesDataset


def main():
    IMAGE_DIR = Path('data')
    jpg = list(IMAGE_DIR.rglob('*.jpg'))
    jpeg = list(IMAGE_DIR.rglob('*.jpeg'))
    png = list(IMAGE_DIR.rglob('*.png'))

    images = []
    images.extend(jpg)
    images.extend(jpeg)
    images.extend(png)
    images.sort()
    data = ImagesDataset(images)
 

    if not images:
        sys.exit('Data directory is empty')
    
    model = nvidia_ssd() 
    model.eval()
    
    if not (Path.cwd() / 'new_data').exists():
        Path("new_data").mkdir(parents=True, exist_ok=True)
   
 
   
    for i, image in tqdm(enumerate(data), ncols=80):

        name, format = data.files[i].name.rsplit('.', 1)
        
        
   
        
        with torch.no_grad():
            detections = model(image)
        results_per_input = processing.decode_results(detections,criteria=0.5, max_output=20)
        convert_and_save(results_per_input, data.files[i], name, format, threshold=0.5)
        #bboxes, classes, confidences = results_per_input[0]
        #n = cv2.imread(str(images[0])) 
        #cv2.imshow('image', n)
        #cv2.waitKey(0)
        #cv2.imshow('asd', 'data/cat1.jpg')
        
 
        


if __name__ == "__main__":
    main()
    
    
   