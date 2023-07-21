import sys
from pathlib import Path

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
    
    
    #main()
