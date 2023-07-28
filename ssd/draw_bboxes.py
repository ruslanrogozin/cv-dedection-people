from pathlib import Path
import cv2


def draw_bboxes(
    prediction,  path_image
):

    if isinstance(path_image, str):
        path_image = Path(path_image)

    original = cv2.imread(str(path_image))
    bboxes, classes, confidences = prediction

    orig_h, orig_w = original.shape[0], original.shape[1]

    for idx in range(len(bboxes)):
        if classes[idx] == 1:
            # get the bounding box coordinates in xyxy format
            x1, y1, x2, y2 = bboxes[idx]
            # resize the bounding boxes from the normalized to 300 pixels
            x1, y1 = int(x1 * 300), int(y1 * 300)
            x2, y2 = int(x2 * 300), int(y2 * 300)
            # resizing again to match the original dimensions of the image
            x1, y1 = int((x1 / 300) * orig_w), int((y1 / 300) * orig_h)
            x2, y2 = int((x2 / 300) * orig_w), int((y2 / 300) * orig_h)
            # draw the bounding boxes around the objects
            cv2.rectangle(original, (x1, y1), (x2, y2),
                          (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(
                original, 'person', (x1, y1+20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 25, 255), 2)

    # cv2.imshow('image', original)
    # cv2.waitKey(0)
    return original
