import os
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from config.config import Configs
from detect_images import detect_image_batch, detect_images_from_folder
from detect_video import detect_one_video, detect_videos_from_folder
from ssd.create_model import nvidia_ssd
from ssd.Detection_model import Detection_model

detection = Detection_model()

app = FastAPI(
    title="Detection people with ssd300",
    description="Using ssd300 for detect people",
)


@app.on_event("startup")
async def startup_event():
    print("Server started ")


def load_image(data):
    return Image.open(BytesIO(data))


@app.post("/detect_image/")
async def detect_image(
    file: UploadFile = File(...),
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
    image = load_image(await file.read())
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    return detect_image_batch(
        model=detection.model,
        images=[img],
        criteria_iou=criteria_iou,
        max_output_iou=max_output_iou,
        prob_threshold=prob_threshold,
        use_head=detection.use_head,
    )["img_num_0"]


@app.post("/detect_video/")
async def detect_video(
    file: UploadFile = File(...),
    batch_size: Annotated[
        int, Path(title="batch_size", gt=0)
    ] = Configs.batch_size,
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
    # https://www.reactfix.com/2022/12/fixed-how-to-upload-list-of-videos.html
    temp = NamedTemporaryFile(delete=False)
    print("f", file.filename)
    extension = file.filename.split(".")[-1] in ("mp4", "avi")
    if not extension:
        raise HTTPException(
            status_code=403, detail="Video must be in mp4 or avi format!"
        )
    try:
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents)
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()
        bbx = detect_one_video(
            model=detection.model,
            video=temp.name,
            device=detection.device,
            batch_size=batch_size,
            criteria_iou=criteria_iou,
            max_output_iou=max_output_iou,
            prob_threshold=prob_threshold,
        )
    except Exception:
        return {"message": "There was an error processing the file"}

    finally:
        os.remove(temp.name)

    return bbx


@app.post("/detect_images_from_folder/{path_to_data}")
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
    ans = detect_images_from_folder(
        model=detection.model,
        device=detection.device,
        path_to_data=path_to_data,
        criteria_iou=criteria_iou,
        max_output_iou=max_output_iou,
        prob_threshold=prob_threshold,
        use_head=detection.use_head,
    )

    return ans


@app.post("/detect_video_from_folder/{path_to_data}")
async def detect_video_from_folder(
    path_to_data: str,
    batch_size: Annotated[
        int, Path(title="batch_size", gt=0)
    ] = Configs.batch_size,
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
    ans = detect_videos_from_folder(
        model=detection.model,
        device=detection.device,
        batch_size=batch_size,
        path_to_data=path_to_data,
        criteria_iou=criteria_iou,
        max_output_iou=max_output_iou,
        prob_threshold=prob_threshold,
        use_head=detection.use_head,
    )

    return ans


@app.put("/model/")
def set_model(
    pretrained_default: bool = False,
    pretrainded_custom: bool = False,
    weight: str = "",
    device: str = "cpu",
    use_head: bool = False,
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


@app.get("/model/info")
def get_model_info():
    return {
        "pretrained_default": detection.pretrained_default,
        "pretrainded_custom": detection.pretrainded_custom,
        "device ": detection.device,
        "label_num": detection.label_num,
        "weight": detection.weight,
    }
