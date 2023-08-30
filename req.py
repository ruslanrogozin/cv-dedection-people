import requests

from utils.utils import draw_bboxes_and_save_image

BASE_URL = "http://127.0.0.1:8000"
input_image_name = "data\\2.jpg"
api_host = "http://127.0.0.1:8000/"
type_rq = "detect_image"

files = {"file": open(input_image_name, "rb")}
params = {
    "criteria_iou": "0.5",
    "max_output_iou": "200",
    "prob_threshold": "0.3",
}


response = requests.post(api_host + type_rq, files=files, params=params)
print(response.json().keys())
d = {}
d[input_image_name] = response.json().pop("img_num_0")
print(d.keys())
draw_bboxes_and_save_image(
    detect_res=d,
    path_new_data="new_www",
    save_image=False,
    show_image=True,
)
