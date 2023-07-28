from pathlib import Path
from ssd.entrypoints import nvidia_ssd
from config.config import Configs
from detect_images import detect_images


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
    model.eval()  # Что происходит при переходи в режим eval.

    detect_images(model=model,
                 configs=configs,
                 work_directory=work_directory)


if __name__ == "__main__":
    main()
