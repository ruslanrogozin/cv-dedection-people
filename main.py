from pathlib import Path

from config.config import Configs
from detect_images import detect_images
from detect_video import detect_video
from ssd.create_model import nvidia_ssd


def main():
    configs = Configs()
    work_directory = Path().cwd()
    device = configs.device

    model = nvidia_ssd(
        pretrained_default=True,
        pretrainded_custom=False,
        path=work_directory / configs.path_weight_model,
        device=device,
    )
    model.eval()

    detect_images(model=model, configs=configs, work_directory=work_directory)

    detect_video(model=model, configs=configs, work_directory=work_directory)


if __name__ == "__main__":
    main()
