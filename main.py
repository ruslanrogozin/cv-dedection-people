from pathlib import Path

import torch

from detect_video import detect_video
from ssd.create_model import nvidia_ssd


def main():
    torch.cuda.empty_cache()
    work_directory = Path().cwd()

    model = nvidia_ssd(
        pretrained_default=True,
        pretrainded_custom=False,
        path=work_directory / "weight",
        device='cuda'

    )


    #model.eval()

    #detect_images(
        #model=model,
        #path_to_data=work_directory / "data",
        #path_new_data=work_directory / "new_data",
   # )

    detect_video(
        model=model,
        device='cuda',
        path_to_data=work_directory / "data",
        path_new_data=work_directory / "new_data",
    )


if __name__ == "__main__":
    main()
