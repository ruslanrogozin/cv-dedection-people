import cv2
import torch
from PIL import Image
from torchvision import transforms

from config.config import Configs
from ssd.decode_results import Processing as processing
from utils.utils import draw_bboxes


def detect_image(
    model,
    images,
    device=Configs.device,
    criteria_iou=Configs.decode_result["criteria"],
    max_output_iou=Configs.decode_result["max_output"],
    prob_threshold=Configs.decode_result["pic_threshold"]
):
    inputs = []
    for image in images:
        img1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img1, "RGB")
        inputs.append(img)

    transform = transforms.Compose(
        [
            #SquarePad(),
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    tensor = torch.stack([transform(img) for img in inputs])

    if device == "cuda" and torch.cuda.is_available():
        tensor = tensor.cuda()

    # tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        detections = model(tensor)

    results_per_inputs = processing.decode_results(
        predictions=detections,
        criteria=criteria_iou,
        max_output=max_output_iou,
    )


    output = []

    for i, results_per_input in enumerate(results_per_inputs):
        best_results_per_input = processing.pick_best(
            detections=results_per_input,
            threshold=prob_threshold,
        )

        new_image = draw_bboxes(
            prediction=best_results_per_input,
            original=images[i],
            use_padding=Configs.use_padding_in_image_transform,
        )
        output.append(new_image)

    return output
