# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNet(nn.Module):
    def __init__(self,  backbone_path=None, weights="IMAGENET1K_V1"):
        super().__init__()

        #backbone == 'resnet50':
        backbone = resnet50(weights=None if backbone_path else weights)
        self.out_channels = [1024, 512, 512, 256, 256, 256]
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)
        #(N, 1024, 38 ,38)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x