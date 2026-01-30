import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class CyrillicLettersDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.classes = [f.name for f in os.scandir('D:/Cyrillic') if f.is_dir()]
        self.img_labels = []
        for root, dirs, files in os.walk('D:/Cyrillic'):
            for file in files:
                self.img_labels.append((self.classes.index(root[-1]), os.path.join(root, file)))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return sum(len(files) for _, _, files in os.walk('D:/Cyrillic'))

    def __getitem__(self, idx):
        img_path = self.img_labels[idx][1]
        image = Image.open(img_path)
        label = self.img_labels[idx][0]
        if self.transform:
            image = self.transform(image)
        return image, label

a = CyrillicLettersDataset('')#transform=transforms.Compose([transforms.ToTensor()])
a[3650][0].show()
print(len(a))
