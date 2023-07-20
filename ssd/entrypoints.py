#https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/ssd/entrypoints.py

import os
import torch
import sys
import urllib.request


def _download_checkpoint(checkpoint):
    ''' load weight ssd300 from url to ssd directory'''
    ckpt_file = os.path.join('ssd', os.path.basename(checkpoint))
    urllib.request.urlretrieve(checkpoint, ckpt_file)
    return ckpt_file

