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



def nvidia_ssd(pretrained=True, **kwargs):
    """Constructs an SSD300 model.
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com
    Args:
        pretrained (bool, True): If True, returns a model pretrained on COCO dataset.
        model_math (str, 'fp32'): returns a model in given precision ('fp32' or 'fp16')
        
    """
    if pretrained:
        #'https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/20.06.0/files/nvidia_ssdpyt_amp_200703.pt'
        checkpoint = 'https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/20.06.0/files/nvidia_ssdpyt_amp_200703.pt'
        model_name = os.path.basename(checkpoint)
        if not os.path.exists('ssd/' + model_name):
            print('---loading model weights and saving to ssd folder--- ')
            ckpt_file = _download_checkpoint(checkpoint)
        else:
            ckpt_file  = os.path.join('ssd', os.path.basename(checkpoint))
            
        print(ckpt_file )
        if not torch.cuda.is_available():
            ckpt = torch.load(ckpt_file),  map_location=torch.device('cpu')
        ckpt = ckpt['model']
        #ckpt = torch.load(ckpt_file)
        #ckpt = ckpt['model']
        #if checkpoint_from_distributed(ckpt):
            #ckpt = unwrap_distributed(ckpt)
        #m.load_state_dict(ckpt)