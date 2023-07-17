import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scripts.convert_and_save import convert_and_save
from scripts.dataloader import ImagesDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm


def main():
    IMAGE_DIR = Path('data')
    jpg = list(IMAGE_DIR.rglob('*.jpg'))
    jpeg = list(IMAGE_DIR.rglob('*.jpeg'))
    png = list(IMAGE_DIR.rglob('*.png'))

    images = []
    images.extend(jpg)
    images.extend(jpeg)
    images.extend(png)

    images.sort()

    if not images:
        sys.exit('Data directory is empty')

    data = ImagesDataset(images)
    # print(data.files)

    model = fasterrcnn_resnet50_fpn(weights=None, progress=False)

    state_dict = torch.load('model\model.DEFAULT')
    model.load_state_dict(state_dict)
    model.eval();  ## Setting Model for Evaluation/Prediction

    if not (Path.cwd() / 'new_data').exists():
        Path("new_data").mkdir(parents=True, exist_ok=True)

    for i, image in tqdm(enumerate(data), ncols=80):

        name, format = data.files[i].name.rsplit('.', 1)
        scale = image[1]

        picture_out = model(image[0])

        if (picture_out[0]['labels'] == 1).any():

            original_image = Image.open(data.files[i])
            original_image = pil_to_tensor(original_image)
            convert_and_save(picture_out, original_image, name, format, scale)
        else:
            continue


if __name__ == "__main__":
    main()
