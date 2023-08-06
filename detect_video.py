from pathlib import Path

import cv2

from detect_one_image import detect_image


def detect_video(
    model,
    configs,
    work_directory,
):
    device = configs.device
    VIDEO_DIR = work_directory / configs.path_data

    mp4 = list(VIDEO_DIR.rglob("*.mp4"))
    avi = list(VIDEO_DIR.rglob("*.avi"))

    videos = []
    videos.extend(mp4)
    videos.extend(avi)

    if len(videos) == 0:
        print("no video found")
        return

    Path(work_directory / configs.path_new_data).mkdir(
        parents=True, exist_ok=True
    )

    for video in videos:
        orginal_name = Path(video).name
        orginal_name = orginal_name.rsplit(".", 1)[0]
        path_save_video = work_directory / configs.path_new_data
        path_save_video = path_save_video / ("new_" + "output.avi")

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

            new_image = detect_image(model=model, image=image, configs=configs)

            cv2.waitKey(0)

            out.write(new_image)

        out.release()
        cv2.destroyAllWindows()
