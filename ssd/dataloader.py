from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
from utils.utils import SquarePad


class ImagesDataset(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """

    def __init__(self, path,
                 device='cpu',
                 transform='DEFAULT',
                 resize_size=300,
                 normalize_mean=(0.5, 0.5, 0.5),
                 normalize_std=(0.5, 0.5, 0.5)
                 ):

        super().__init__()
        self.device = device
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.resize_size = resize_size

        IMAGE_DIR = Path(path)
        jpg = list(IMAGE_DIR.rglob("*.jpg"))
        jpeg = list(IMAGE_DIR.rglob("*.jpeg"))
        png = list(IMAGE_DIR.rglob("*.png"))
        images = []
        images.extend(jpg)
        images.extend(jpeg)
        images.extend(png)
        assert len(images) != 0, 'no images found'

        self.images = images
        self._len = len(self.images)

        self.transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(self.resize_size),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean,
                                 std=self.normalize_std)])
        # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        """return torch.tensor(image) and shape original image"""
        file = self.images[index]
        tensor = Image.open(file).convert('RGB')
        tensor = self.transform(tensor)
        if self.device == 'cuda' and torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor, file
