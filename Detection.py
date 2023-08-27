from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from config.config import Configs
from ssd.create_model import nvidia_ssd
from ssd.dataloader import ImagesDataset
from ssd.decode_results import Processing as processing


class Detection:
    def __init__(
        self,
        work_directory=Path().cwd(),
        device=Configs.device,
        label_num=81,
        pretrained_default=True,
        pretrainded_custom=False,
        path_weight_model=Configs.path_weight_model,
        use_head=False,
    ):
        self.work_directory = work_directory
        self.device = device
        self.label_num = label_num
        self.pretrained_default = pretrained_default
        self.pretrainded_custom = pretrainded_custom
        model = nvidia_ssd(
            pretrained_default=self.pretrained_default,
            pretrainded_custom=self.pretrainded_custom,
            path=self.work_directory / path_weight_model,
            device=self.device,
            label_num=self.label_num,
        )
        self.info = {1: "person", 2: "head"}
        self.model = model
        self.weight = "default"
        self.use_head = use_head

    def detect_and_save_image(
        self,
        path_to_data=Configs.path_data,
        criteria_iou=Configs.decode_result["criteria"],
        max_output_iou=Configs.decode_result["max_output"],
        prob_threshold=Configs.decode_result["pic_threshold"],
        path_new_data=Configs.path_new_data,
        use_head = False
    ):
        ans = self.detect_image_from_folder(
            path_to_data, criteria_iou, max_output_iou, prob_threshold
        )
        if ans == "no images found!":
            return "no images found!"
        if isinstance(path_new_data, str):
            path_new_data = Path(path_new_data)
        path_new_data.mkdir(parents=True, exist_ok=True)
        info = {"person": (0, 0, 255), "head": (0, 255, 0)}
        for img in ans.keys():
            original = cv2.imread(str(img))
            bbx_ps = ans[img]["person"]
            for bbx_p in bbx_ps:
                x1, y1, x2, y2 = bbx_p

                cv2.rectangle(
                    original, (x1, y1), (x2, y2), info["person"], 2, cv2.LINE_AA
                )
                cv2.putText(
                    original,
                    "person",
                    (x1, y1 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    info["person"],
                    1,
                )
            if use_head and ans[img]["head"]:
                bbx_hs = ans[img]["head"]
                for bbx_h in bbx_hs:
                    x1, y1, x2, y2 = bbx_h
                    cv2.rectangle(
                        original, (x1, y1), (x2, y2), info["head"], 2, cv2.LINE_AA
                    )
                    cv2.putText(
                        original,
                        "head",
                        (x1, y1 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        info["head"],
                        1,
                    )
            orginal_name = img.name
            path_save_image = path_new_data / ("new_" + orginal_name)
            cv2.imwrite(str(path_save_image), original)



    def detect_image_from_folder(
        self,
        path_to_data=Configs.path_data,
        criteria_iou=Configs.decode_result["criteria"],
        max_output_iou=Configs.decode_result["max_output"],
        prob_threshold=Configs.decode_result["pic_threshold"],
    ):
        path_to_data = Path(path_to_data)
        data = ImagesDataset(
            path=path_to_data,
            device=self.device,
        )
        print("run detect images")
        if len(data.images) == 0:
            return "no images found!"

        self.model.eval()
        ans = {}
        for image, file in tqdm(data):
            ans.setdefault(file, {})
            image = image.unsqueeze(0)
            if self.device == "cuda" and torch.cuda.is_available():
                image = image.cuda()

            with torch.no_grad():
                detections = self.model(image)

            results_per_input = processing.decode_results(
                predictions=detections,
                criteria=criteria_iou,
                max_output=max_output_iou,
            )

            best_results_per_input = processing.pick_best(
                detections=results_per_input[0],
                threshold=prob_threshold,
            )

            original = cv2.imread(str(file))

            orig_h, orig_w = original.shape[0], original.shape[1]
            bboxes, classes, _ = best_results_per_input

            ans[file]["person"] = []
            if self.use_head:
                ans[file]["head"] = []

            for idx, bbox in enumerate(bboxes):
                if not self.use_head and (classes[idx] != 1):
                    continue
                x1, y1, x2, y2 = bbox
                # resize the bounding boxes from the normalized to 300 pixels
                x1, y1 = int(x1 * 300), int(y1 * 300)
                x2, y2 = int(x2 * 300), int(y2 * 300)
                # resizing again to match the original dimensions of the image
                x1, y1 = int((x1 / 300) * orig_w), int((y1 / 300) * orig_h)
                x2, y2 = int((x2 / 300) * orig_w), int((y2 / 300) * orig_h)
                # draw the bounding boxes around the objects
                info = self.info[classes[idx]]
                ans[file][info].append([x1, y1, x2, y2])

        return ans
