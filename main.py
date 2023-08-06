from pathlib import Path

from config.config import Configs
from detect_images import detect_images
from detect_video import detect_video
from ssd.create_model import nvidia_ssd


def main():
    configs = Configs()
    work_directory = Path().cwd()
    device = "cpu"

    model = nvidia_ssd(
        pretrained_default=True,
        pretrainded_custom=False,
        path=work_directory/ configs.path_weight_model,
        device=device,
    )
    model.eval()

    detect_images(
        model=model,
        device=device,
        path_to_data=work_directory/ "data",
        path_new_data=work_directory / "new_data",
    )

    detect_video(model=model, configs=configs, work_directory=work_directory)


if __name__ == "__main__":
    main()
