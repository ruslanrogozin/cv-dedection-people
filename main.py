from pathlib import Path
from tqdm import tqdm
import torch

import cv2

from ssd.draw_bboxes import draw_bboxes
from ssd.entrypoints import nvidia_ssd
from ssd.nvidia_ssd_processing_utils import Processing as processing
from ssd.dataloader import ImagesDataset
from config.config import Configs


def main():
    configs = Configs()
    device = configs.device
    work_directory = Path().cwd()  # Path(__file__)

    data = ImagesDataset(
        path=work_directory / configs.path_data,
        device=device,
        transform='DEFAULT',
        resize_size=configs.image_loader_params['resize_size'],
        center_crop_cize=configs.image_loader_params['center_crop_cize'],
        normalize_mean=configs.image_loader_params['normalize_mean'],
        normalize_std=configs.image_loader_params['normalize_std']
    )

    model = nvidia_ssd(
        pretrained_default=True,
        pretrainded_custom=False,
        path=work_directory / configs.path_weight_model,
        device=device
    )
    model.eval()  # Что происходит при переходи в режим eval.

    Path(work_directory /
         configs.path_new_data).mkdir(parents=True, exist_ok=True)

    for image,  file in tqdm(data):

        image = image.unsqueeze(0)
        with torch.no_grad():  # нужен ли no grad при eval режим
            detections = model(image)

        results_per_input = processing.decode_results(
            predictions=detections,
            criteria=configs.decode_result['criteria'],
            max_output=configs.decode_result['max_output'])

        best_results_per_input = processing.pick_best(
            detections=results_per_input[0],
            threshold=configs.decode_result['pic_threshold'])

        # Вызвать prepare tensor nwhc
        # в даталоадер. path.stem path.suffix. В одно название
        # name, format = data.images[i].name.rsplit(".", 1)
       # image = image.unsqueeze(0)
        # with torch.no_grad():  # нужен ли no grad при eval режим
        # transform images
        #  detections = model(image)
        # results_per_input = processing.decode_results(
        # detections, criteria=0.8, max_output=20
        # )  # В конифиги

        new_image = draw_bboxes(
            best_results_per_input, file)

        orginal_name = Path(file).name
        path_save_image = work_directory / configs.path_new_data
        path_save_image = path_save_image / ('new_' + orginal_name)
        #print(path_save_image )
        cv2.imwrite(str(path_save_image ), new_image)

        # )  # В конифиги
        # добавить сигнатуру функции
        # проверить чтобы результат на исходном изображении был
        # Сохранять даже если ничего не нашел


if __name__ == "__main__":
    main()
