from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import pil_loader


class ImagesDataset(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """

    def __init__(self, files, transform=None, rescale_size=600):
        super().__init__()
        self.files = sorted(files)
        self.rescale_size = rescale_size
        self._len = len(self.files)
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size=(self.rescale_size, self.rescale_size)),
                transforms.ToTensor()])

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        ''' return torch.tensor(image) and shape original image'''
        file = self.files[index]

        x = pil_loader(file)
        width, height = x.size
        x_scale = self.rescale_size / width
        y_scale = self.rescale_size / height
        x = self.transform(x)
        return x.unsqueeze(dim=0), (x_scale, y_scale)
