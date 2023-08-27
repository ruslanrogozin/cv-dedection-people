from pathlib import Path
from typing import Annotated

import cv2
import torch
from fastapi import FastAPI, HTTPException

from config.config import Configs
from ssd.create_model import nvidia_ssd
from ssd.dataloader import ImagesDataset
from ssd.decode_results import Processing as processing


class Detection:
    def __init__(
        self,
    ):
        self.work_directory = Path().cwd()
        self.device = Configs.device
        self.label_num = 81
        self.pretrained_default = True
        self.pretrainded_custom = False
        model = nvidia_ssd(
            pretrained_default=self.pretrained_default,
            pretrainded_custom=self.pretrainded_custom,
            path=self.work_directory / "weight",
            device=self.device,
            label_num=self.label_num,
        )
        self.info = {1: "person", 2: "head"}
        self.model = model
        self.weight = "default"
        self.use_head = False

    def detect_image_from_folder(
        self,
        path_to_data=Configs.path_data,
        criteria_iou=Configs.decode_result["criteria"],
        max_output_iou=Configs.decode_result["max_output"],
        prob_threshold=Configs.decode_result["pic_threshold"],
    ):
        path_to_data = Path(path_to_data)
        data = ImagesDataset(
            path=path_to_data,
            device=self.device,
        )

        if len(data.images) == 0:
            return "no images found!"

        self.model.eval()
        ans = {}
        for image, file in data:
            ans.setdefault(file, {})
            image = image.unsqueeze(0)
            if self.device == "cuda" and torch.cuda.is_available():
                image = image.cuda()

            with torch.no_grad():
                detections = self.model(image)

            results_per_input = processing.decode_results(
                predictions=detections,
                criteria=criteria_iou,
                max_output=max_output_iou,
            )

            best_results_per_input = processing.pick_best(
                detections=results_per_input[0],
                threshold=prob_threshold,
            )

            original = cv2.imread(str(file))

            orig_h, orig_w = original.shape[0], original.shape[1]
            bboxes, classes, _ = best_results_per_input

            ans[file]["person"] = []
            if self.use_head:
                ans[file]["head"] = []

            for idx, bbox in enumerate(bboxes):
                if not self.use_head and (classes[idx] != 1):
                    continue
                x1, y1, x2, y2 = bbox
                # resize the bounding boxes from the normalized to 300 pixels
                x1, y1 = int(x1 * 300), int(y1 * 300)
                x2, y2 = int(x2 * 300), int(y2 * 300)
                # resizing again to match the original dimensions of the image
                x1, y1 = int((x1 / 300) * orig_w), int((y1 / 300) * orig_h)
                x2, y2 = int((x2 / 300) * orig_w), int((y2 / 300) * orig_h)
                # draw the bounding boxes around the objects
                info = self.info[classes[idx]]
                ans[file][info].append([x1, y1, x2, y2])

        return ans


detection = Detection()

app = FastAPI(
    title="Detection people with ssd300",
    description="Using ssd300 for detect people",
)


@app.on_event("startup")
async def startup_event():
    print("Server started ")
    app.package = {"model": detection.model}


@app.post("/users/{user_id}")
def change_user_name(user_id: int):
    return {"status": 200, "data": app.package["model"]}


@app.post("/dect_from_folder/{path_to_data}")
async def detect_image_from_folder(
    path_to_data: str,
    criteria_iou: Annotated[
        float, Path(title="riteria_iou", gt=0, le=1)
    ] = Configs.decode_result["criteria"],
    max_output_iou: Annotated[
        int, Path(title="max_output_iou", gt=0)
    ] = Configs.decode_result["max_output"],
    prob_threshold: Annotated[
        float, Path(title="prob_threshold", ge=0, le=1)
    ] = Configs.decode_result["pic_threshold"],
):
    ans = detection.detect_image_from_folder(
        path_to_data=path_to_data,
        criteria_iou=criteria_iou,
        max_output_iou=max_output_iou,
        prob_threshold=prob_threshold,
    )

    return ans


@app.put("/model/")
def edit_person(
    pretrained_default: bool = False,
    pretrainded_custom: bool = False,
    weight: str = "",
    device: str = "cpu",
    use_head : bool = False
):
    if pretrained_default:
        detection.pretrained_default = True
        return {"status": "default model"}
    if pretrainded_custom:
        work_directory = detection.work_directory
        if not weight:
            raise HTTPException(status_code=404, detail="Weight not found!")
        model = nvidia_ssd(
            pretrained_default=False,
            pretrainded_custom=True,
            path=work_directory / weight,
            device=device,
            label_num=3,
        )
        detection.model = model
        detection.device = device
        detection.pretrained_default = False
        detection.pretrainded_custom = True
        detection.label_num = 3
        detection.weight = weight
        detection.use_head = use_head

        return {"status": "set pretrained model"}
    return {"status": "default model"}


@app.put("/model/info")
def get_model_info():
    return {
        "pretrained_default": detection.pretrained_default,
        "pretrainded_custom": detection.pretrainded_custom,
        "device ": detection.device,
        "label_num": detection.label_num,
        "weight": detection.weight,
    }
