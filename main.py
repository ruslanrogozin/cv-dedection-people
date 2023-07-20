import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchsummary import summary



from ssd.model import ResNet

def main():
    return 0
 
        


if __name__ == "__main__":
    print(1)
    IMAGE_SIZE = (300, 300)
    backbone = ResNet()

    print(summary(backbone, (3, *IMAGE_SIZE)))
    
    #main()
