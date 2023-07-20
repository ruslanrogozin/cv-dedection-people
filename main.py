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

    url = 'https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/20.06.0/files/nvidia_ssdpyt_amp_200703.pt'
    print(os.path.exists('ssd/nvidia_ssdpyt_amp_200703.pt'))
    nvidia_ssd(True)
   
    
    
    
    
    
    #main()
