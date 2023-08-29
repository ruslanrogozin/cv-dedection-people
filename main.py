from pathlib import Path

import torch

from detect_images_from_folder import detect_images
from ssd.create_model import nvidia_ssd
from utils.utils import draw_bboxes_and_save


def main():
    torch.cuda.empty_cache()
    work_directory = Path().cwd()

    model = nvidia_ssd(
        pretrained_default=False,
        pretrainded_custom=True,
        path=work_directory / "weight\\state best_model_at_adam2.pth",
        device="cuda",
        label_num=3,
    )
    res = detect_images(
        model=model,
        device="cuda",
        path_to_data=work_directory / "data",
        prob_threshold=0.3,
        use_head=True,
    )
    draw_bboxes_and_save(res, save_image=True, use_head=True)


# detect_video(
# model=model,
# device="cuda",
# path_to_data=work_directory / "data",
# path_new_data=work_directory / "new_data",
# prob_threshold=0.3,
# use_head=False,
# )


if __name__ == "__main__":
    main()
