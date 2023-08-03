from pathlib import Path
from tqdm import tqdm
import torch
import cv2

from ssd.nvidia_ssd_processing_utils import Processing as processing
from ssd.dataloader import ImagesDataset
from utils.utils import draw_bboxes


def detect_images(model, configs, work_directory):
    device = configs.device

    data = ImagesDataset(
        path=work_directory / configs.path_data,
        device=device,
        resize_size=configs.image_loader_params["resize_size"],
        normalize_mean=configs.image_loader_params["normalize_mean"],
        normalize_std=configs.image_loader_params["normalize_std"],
    )

    Path(work_directory / configs.path_new_data).mkdir(parents=True, exist_ok=True)

    for image, file in tqdm(data):

        image = image.unsqueeze(0)
        with torch.no_grad():
            detections = model(image)

        results_per_input = processing.decode_results(
            predictions=detections,
            criteria=configs.decode_result["criteria"],
            max_output=configs.decode_result["max_output"],
        )

        best_results_per_input = processing.pick_best(
            detections=results_per_input[0], threshold=configs.decode_result["pic_threshold"]
        )

        new_image = draw_bboxes(
            prediction=best_results_per_input, original=file, use_padding=configs.use_padding_in_image_transform
        )

        orginal_name = Path(file).name
        path_save_image = work_directory / configs.path_new_data
        path_save_image = path_save_image / ("new_" + orginal_name)
        cv2.imwrite(str(path_save_image), new_image)
