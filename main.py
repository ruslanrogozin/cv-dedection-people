import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchsummary import summary
import os


from ssd.model import ResNet
from ssd.entrypoints import _download_checkpoint, nvidia_ssd

def main():
    return 0
 
        


if __name__ == "__main__":
    IMAGE_SIZE = (300, 300)
    backbone = ResNet()
    
    model = nvidia_ssd()

   
    
    
    
    
    
    #main()
