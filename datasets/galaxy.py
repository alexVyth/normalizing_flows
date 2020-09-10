import cv2
from torch.utils.data import Dataset
from astroNN.datasets import galaxy10


class Galaxy(Dataset):
    def __init__(self, root, train=True, transform=None, mini_data_size=None):
        self.transform = transform
        self.imgs, self.labels = galaxy10.load_data()

    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = cv2.resize(img, (32, 32))
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return self.imgs.shape[0]
