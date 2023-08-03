import numpy as np
from .utils_ssd300 import dboxes300_coco, Encoder


class Processing:
    @staticmethod
    def decode_results(predictions, criteria=0.5, max_output=20):
        dboxes = dboxes300_coco()
        encoder = Encoder(dboxes)
        ploc, plabel = [val.float() for val in predictions]
        results = encoder.decode_batch(ploc, plabel, criteria=criteria, max_output=max_output)
        return [[pred.detach().cpu().numpy() for pred in detections] for detections in results]

    @staticmethod
    def pick_best(detections, threshold=0.3):
        bboxes, classes, confidences = detections
        best = np.argwhere(confidences > threshold)[:, 0]
        return [pred[best] for pred in detections]

    @staticmethod
    def get_coco_object_dictionary():
        import os

        file_with_coco_names = "category_names.txt"

        if not os.path.exists(file_with_coco_names):
            print("Downloading COCO annotations.")
            import urllib
            import zipfile
            import json
            import shutil

            urllib.request.urlretrieve(
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                "cocoanno.zip",
            )
            with zipfile.ZipFile("cocoanno.zip", "r") as f:
                f.extractall()
            print("Downloading finished.")
            with open("annotations/instances_val2017.json", "r") as COCO:
                js = json.loads(COCO.read())
            class_names = [category["name"] for category in js["categories"]]
            open("category_names.txt", "w").writelines([c + "\n" for c in class_names])
            os.remove("cocoanno.zip")
            shutil.rmtree("annotations")
        else:
            class_names = open("category_names.txt").readlines()
            class_names = [c.strip() for c in class_names]
        return class_names
