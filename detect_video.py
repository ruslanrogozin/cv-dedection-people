from pathlib import Path

import cv2

from config.config import Configs
from detect_one_image import detect_image


def detect_video(
    model,
    device=Configs.device,
    path_to_data=Configs.path_data,
    path_new_data=Configs.path_new_data,
):
    if isinstance(path_to_data, str):
        path_to_data = Path(path_to_data)
    if isinstance(path_new_data, str):
        path_new_data = Path(path_new_data)

    mp4 = list(path_to_data.rglob("*.mp4"))
    avi = list(path_to_data.rglob("*.avi"))

    videos = []
    videos.extend(mp4)
    videos.extend(avi)

    if len(videos) == 0:
        print("no video found")
        return

    path_new_data.mkdir(parents=True, exist_ok=True)

    for video in videos:
        orginal_name = video.name
        orginal_name = orginal_name.rsplit(".", 1)[0]
        path_save_video = path_new_data

        path_save_video = path_save_video / ("new_" + orginal_name + ".avi")

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)  # fps

        f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            str(path_save_video), fourcc, fps, (f_width, f_height)
        )

        while True:
            ret, image = cap.read()

            if not ret:
                break

            new_image = detect_image(model=model, device=device, image=image)

            cv2.waitKey(0)

            out.write(new_image)

        out.release()
        cv2.destroyAllWindows()
