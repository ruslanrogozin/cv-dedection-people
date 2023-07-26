import sys

from pathlib import Path
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

from ssd.convert_and_save import convert_and_save
from ssd.entrypoints import nvidia_ssd
from ssd.nvidia_ssd_processing_utils import Processing as processing
from ssd.dataloader import ImagesDataset
#from utils.utils import Normalize


def main():

    #import os
    #print(os.getcwd())
    #print(__file__)
    p = Path(__file__)
    print(p.parents[0], p.parents[1])
    print(p.parents[0] / 'data')


    data = ImagesDataset(path = 'data',
                         transform = 'DEFAULT')
    import os

    #print(os.path.basename(data.images[0]))
    #print(os.path.splitext(data.images[0]))


    #sys.exit("aaaaaaaaaaaaaaaaaaaaaaaa")
    #sys.exit("aaaaaaaaaaaaaaaaaaaaaaaa")

    # добавить выбор device cpu/cuda
    # какие веса используются передать
    model = nvidia_ssd()
    model.eval()  # Что происходит при переходи в режим eval.

    # Path.cwd() / 'new_data' == Path('new_data')
    # Зафикисровать путь от файла и вынести в конфиги / убрать проверку
    #$if not (Path.cwd() / "new_data").exists():
    Path("new_data").mkdir(parents=True, exist_ok=True)

    for i, image in tqdm(enumerate(data), ncols=20):
        # Вызвать prepare tensor nwhc
        # в даталоадер. path.stem path.suffix. В одно название
        name, format = data.images[i].name.rsplit(".", 1)
        image = image.unsqueeze(0)
        with torch.no_grad():  # нужен ли no grad при eval режим
            # transform images
            detections = model(image)
        results_per_input = processing.decode_results(
            detections, criteria=0.8, max_output=20
        )  # В конифиги
        convert_and_save(
            results_per_input, data.images[i], name, format, threshold=0.55
        )  # В конифиги
        # добавить сигнатуру функции
        # проверить чтобы результат на исходном изображении был
        # Сохранять даже если ничего не нашел


if __name__ == "__main__":
    main()
