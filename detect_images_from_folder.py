from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from config.config import Configs
from ssd.dataloader import ImagesDataset
from ssd.decode_results import Processing as processing
from utils.utils import draw_bboxes


def detect_images(
    model,
    device=Configs.device,
    path_to_data=Configs.path_data,
    path_new_data=Configs.path_new_data,
    criteria_iou=Configs.decode_result["criteria"],
    max_output_iou=Configs.decode_result["max_output"],
    prob_threshold=Configs.decode_result["pic_threshold"],
):
    print("run detect images")
    model.eval()
    if isinstance(path_to_data, str):
        path_to_data = Path(path_to_data)
    if isinstance(path_new_data, str):
        path_new_data = Path(path_new_data)

    data = ImagesDataset(
        path=path_to_data,
        device=device,
    )

    if len(data.images) == 0:
        print("no images found!")
        return

    path_new_data.mkdir(parents=True, exist_ok=True)

    for image, file in tqdm(data):
        image = image.unsqueeze(0)
        if device == "cuda" and torch.cuda.is_available():
            image = image.cuda()

        with torch.no_grad():
            detections = model(image)

        results_per_input = processing.decode_results(
            predictions=detections,
            criteria=criteria_iou,
            max_output=max_output_iou,
        )

        best_results_per_input = processing.pick_best(
            detections=results_per_input[0],
            threshold=prob_threshold,
        )

        new_image = draw_bboxes(
            prediction=best_results_per_input,
            original=file,
            use_padding=Configs.use_padding_in_image_transform,
        )

        orginal_name = file.name
        path_save_image = path_new_data
        path_save_image = path_save_image / ("new_" + orginal_name)
        cv2.imwrite(str(path_save_image), new_image)
