from pathlib import Path

import cv2
from tqdm import tqdm

from config.config import Configs
from detect_batch_image import detect_image


def detect_video(
    model,
    device=Configs.device,
    path_to_data=Configs.path_data,
    path_new_data=Configs.path_new_data,
    batch_size=Configs.batch_size,
    criteria_iou=Configs.decode_result["criteria"],
    max_output_iou=Configs.decode_result["max_output"],
    prob_threshold=Configs.decode_result["pic_threshold"],
    use_head=Configs.use_head,
    use_padding_in_transform=Configs.use_padding_in_image_transform,
):
    print("run detect video")
    model.eval()
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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            str(path_save_video), fourcc, fps, (f_width, f_height)
        )
        batch_number = total_frames // batch_size + int(
            (total_frames % batch_size) > 0
        )

        pbar = tqdm(
            total=batch_number,
            desc="extracting frames at fps: {}".format(fps),
        )
        count = 0
        batch = []
        while cap.isOpened():
            ret, image = cap.read()

            if not ret:
                break
            batch.append(image)

            if (len(batch) == batch_size) or (
                (total_frames % batch_size == len(batch))
                and (count == batch_number - 1)
            ):
                new_images = detect_image(
                    model=model,
                    device=device,
                    images=batch,
                    criteria_iou=criteria_iou,
                    max_output_iou=max_output_iou,
                    prob_threshold=prob_threshold,
                    use_padding_in_transform=use_padding_in_transform,
                    use_head=use_head,
                )
                for new_image in new_images:
                    out.write(new_image)

                batch = []
                pbar.update(1)
                count += 1

        pbar.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
