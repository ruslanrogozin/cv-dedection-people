import numpy as np

from .utils_ssd300 import Encoder, dboxes300_coco


class Processing:
    @staticmethod
    def decode_results(predictions, criteria=0.5, max_output=20):
        dboxes = dboxes300_coco()
        encoder = Encoder(dboxes)
        ploc, plabel = [val.float() for val in predictions]
        results = encoder.decode_batch(
            ploc, plabel, criteria=criteria, max_output=max_output
        )
        return [
            [pred.detach().cpu().numpy() for pred in detections]
            for detections in results
        ]

    @staticmethod
    def pick_best(detections, threshold=0.3):
        bboxes, classes, confidences = detections
        best = np.argwhere(confidences > threshold)[:, 0]
        return [pred[best] for pred in detections]
