import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
from ssd.nvidia_ssd_processing_utils import Processing as processing

def convert_and_save(prediction, original_image, name, format_data,  threshold=0.4, path='new_data/'):
    ''' function for convetr and save pictures'''

    best_results_per_input = [processing.pick_best(results, threshold) for results in prediction]
    print(best_results_per_input)
