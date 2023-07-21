import numpy as np
import torch
import skimage
from skimage import io, transform

from .utils import dboxes300_coco, Encoder

class Processing:
        @staticmethod
        def load_image(image_path):
            """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
            img = skimage.img_as_float(io.imread(image_path))
            if len(img.shape) == 2:
                img = np.array([img, img, img]).swapaxes(0, 2)
            return img

        @staticmethod
        def rescale(img, input_height, input_width):
            """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
            aspect = img.shape[1] / float(img.shape[0])
            if (aspect > 1):
                # landscape orientation - wide image
                res = int(aspect * input_height)
                imgScaled = transform.resize(img, (input_width, res))
            if (aspect < 1):
                # portrait orientation - tall image
                res = int(input_width / aspect)
                imgScaled = transform.resize(img, (res, input_height))
            if (aspect == 1):
                imgScaled = transform.resize(img, (input_width, input_height))
            return imgScaled

        @staticmethod
        def crop_center(img, cropx, cropy):
            """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
            y, x, c = img.shape
            startx = x // 2 - (cropx // 2)
            starty = y // 2 - (cropy // 2)
            return img[starty:starty + cropy, startx:startx + cropx]

        @staticmethod
        def normalize(img, mean=128, std=128):
            img = (img * 256 - mean) / std
            return img

        @staticmethod
        def prepare_tensor(inputs, fp16=False):
            NHWC = np.array(inputs)
            if len(NHWC.shape) < 4:
                NHWC = np.expand_dims(NHWC,0)
                
            NCHW = np.swapaxes(np.swapaxes(NHWC, 1, 3), 2, 3)
            tensor = torch.from_numpy(NCHW)
            tensor = tensor.contiguous()
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            tensor = tensor.float()
            if fp16:
                tensor = tensor.half()
            return tensor

        @staticmethod
        def prepare_input(img_uri):
            img = Processing.load_image(img_uri)
            #original_shape = img.shape
            img = Processing.rescale(img, 300, 300)
            img = Processing.crop_center(img, 300, 300)
            img = Processing.normalize(img)
            return img#, original_shape

        @staticmethod
        def decode_results(predictions):
            dboxes = dboxes300_coco()
            encoder = Encoder(dboxes)
            ploc, plabel = [val.float() for val in predictions]
            results = encoder.decode_batch(ploc, plabel, criteria=0.5, max_output=20)
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
                urllib.request.urlretrieve("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "cocoanno.zip")
                with zipfile.ZipFile("cocoanno.zip", "r") as f:
                    f.extractall()
                print("Downloading finished.")
                with open("annotations/instances_val2017.json", 'r') as COCO:
                    js = json.loads(COCO.read())
                class_names = [category['name'] for category in js['categories']]
                open("category_names.txt", 'w').writelines([c+"\n" for c in class_names])
                os.remove("cocoanno.zip")
                shutil.rmtree("annotations")
            else:
                class_names = open("category_names.txt").readlines()
                class_names = [c.strip() for c in class_names]
            return class_names
        