import sys
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torchsummary import summary
import os

# import cv2

# pre-commit hooks
from ssd.model import ResNet
from ssd.convert_and_save import convert_and_save
from ssd.entrypoints import _download_checkpoint, nvidia_ssd
from ssd.nvidia_ssd_processing_utils import Processing as processing
from ssd.dataloader import ImagesDataset


def main():
    # в файл с конфигами и там же определим текущую папку Config.image_dir
    IMAGE_DIR = Path("data")
    jpg = list(IMAGE_DIR.rglob("*.jpg"))
    jpeg = list(IMAGE_DIR.rglob("*.jpeg"))
    png = list(IMAGE_DIR.rglob("*.png"))

    images = []
    images.extend(jpg)
    images.extend(jpeg)
    images.extend(png)
    images.sort()  # сортировка двойная
    # torcvision.Compose(
    #                 torcvision.rescale(),
    #                 crop_center.norm()
    #             )
    # передать в image dataset
    # отсюда не очивидно, что происходит сортировка и трансформы
    data = ImagesDataset(images)

    if not images:
        sys.exit("Data directory is empty")  # выше по логике

    # добавить выбор device cpu/cuda
    # какие веса используются передать
    model = nvidia_ssd()
    model.eval()  # Что происходит при переходи в режим eval.

    # Path.cwd() / 'new_data' == Path('new_data')
    # Зафикисровать путь от файла и вынести в конфиги / убрать проверку
    if not (Path.cwd() / "new_data").exists():
        Path("new_data").mkdir(parents=True, exist_ok=True)

    for i, image in tqdm(enumerate(data), ncols=80):
        # Вызвать prepare tensor nwhc
        # в даталоадер. path.stem path.suffix. В одно название
        name, format = data.files[i].name.rsplit(".", 1)
        with torch.no_grad():  # нужен ли no grad при eval режим
            # transform images
            detections = model(image)
        results_per_input = processing.decode_results(
            detections, criteria=0.8, max_output=20
        )  # В конифиги
        convert_and_save(
            results_per_input, data.files[i], name, format, threshold=0.5
        )  # В конифиги
        # добавить сигнатуру функции
        # проверить чтобы результат на исходном изображении был
        # Сохранять даже если ничего не нашел


if __name__ == "__main__":
    main()
