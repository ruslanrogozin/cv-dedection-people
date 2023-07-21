from torch.utils.data import Dataset
from torchvision import transforms
from ssd.nvidia_ssd_processing_utils import Processing as processing


class ImagesDataset(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """

    def __init__(self, files):
        super().__init__()
        self.files = sorted(files)
        
    def __len__(self):
        return self._len

    def __getitem__(self, index):
        ''' return torch.tensor(image) and shape original image'''
        file = self.files[index]
        inputs = processing.prepare_input(file) #HWC 
        tensor = processing.prepare_tensor(inputs)
        return tensor 
