import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class CUHK03Dataset(Dataset):
    def __init__(self, data, image_folder_path, transform=None):
        self.data = data
        self.image_folder_path = image_folder_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1 = Image.open(
            os.path.join(self.image_folder_path, self.data["image1"][idx])
        )
        img2 = Image.open(
            os.path.join(self.image_folder_path, self.data["image2"][idx])
        )

        label = self.data["label"][idx]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)
