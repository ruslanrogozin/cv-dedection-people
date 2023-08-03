from torchvision import transforms
import torch
from PIL import Image
import cv2
from utils.utils import SquarePad, draw_bboxes

from ssd.nvidia_ssd_processing_utils import Processing as processing


def detect_image(model, image, configs):

    image = image.copy()
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img1, "RGB")

    resize_size = configs.image_loader_params["resize_size"]
    normalize_mean = configs.image_loader_params["normalize_mean"]
    normalize_std = configs.image_loader_params["normalize_std"]

    transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize(resize_size),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )

    tensor = transform(img)

    if configs.device == "cuda" and torch.cuda.is_available():
        tensor = tensor.cuda()

    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        detections = model(tensor)

    results_per_input = processing.decode_results(
        predictions=detections,
        criteria=configs.decode_result["criteria"],
        max_output=configs.decode_result["max_output"],
    )

    best_results_per_input = processing.pick_best(
        detections=results_per_input[0], threshold=configs.decode_result["pic_threshold"]
    )

    new_image = draw_bboxes(prediction=best_results_per_input, original=image, use_padding=True)

    return new_image
