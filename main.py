from pathlib import Path
from ssd.create_model import nvidia_ssd
from config.config import Configs
from detect_images import detect_images
from detect_video import detect_video


def main():
    configs = Configs()
    work_directory = Path().cwd()
    device = configs.device

    model = nvidia_ssd(
        pretrained_default=True,
        pretrainded_custom=False,
        path=work_directory / configs.path_weight_model,
        device=device
    )
    model.eval()

    try:
        detect_images(model=model,
                      configs=configs,
                      work_directory=work_directory)
    except Exception:
        print('no images found!')

    try:
        detect_video(model=model,
                     configs=configs,
                     work_directory=work_directory)
    except Exception:
        print('no video found!')


if __name__ == "__main__":
    main()
