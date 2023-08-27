from pathlib import Path

import torch

from detect_images_from_folder import detect_images
from ssd.create_model import nvidia_ssd


def main():
    torch.cuda.empty_cache()
    work_directory = Path().cwd()
    #detection = Detection(
    #    pretrained_default=False,
      #  pretrainded_custom=True,
       # path_weight_model="weight\\state best_model_at_adam2.pth",
      # label_num=3,
       # use_head=True
    #)

    #detection.model.eval()

   # detection.detect_and_save_image(
       # path_to_data=work_directory / "data",
       # prob_threshold= 0.2,
       # criteria_iou = 0.4,
        #use_head=False
       #)
    model = nvidia_ssd(
        pretrained_default=False,
        pretrainded_custom=True,
        path=work_directory / "weight\\state best_model_at_adam2.pth",
        device="cuda",
        label_num=3
    )
    detect_images(
        model=model,
        device="cuda",
        path_to_data=work_directory / "data",
        path_new_data=work_directory / "new_data",
        prob_threshold=0.3,
        use_head=True,
    )


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
