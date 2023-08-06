from pathlib import Path

from detect_images import detect_images
from detect_video import detect_video
from ssd.create_model import nvidia_ssd


def main():
    work_directory = Path().cwd()

    model = nvidia_ssd(
        pretrained_default=True,
        pretrainded_custom=False,
        path=work_directory / "weight",
    )

    model.eval()

    detect_images(
        model=model,
        path_to_data=work_directory / "data",
        path_new_data=work_directory / "new_data",
    )

    detect_video(
        model=model,
        path_to_data=work_directory / "data",
        path_new_data=work_directory / "new_data",
    )


if __name__ == "__main__":
    main()
