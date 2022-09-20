import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from PIL import Image

# Need to override __init__, __len__, __getitem__
# as per datasets requirement
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, labelsFile, rootDir, sourceTransform):
        self.data = pd.read_csv(labelsFile)
        self.rootDir = rootDir
        self.sourceTransform = sourceTransform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imagePath = self.rootDir + "/" + self.data['photo_x'][idx]
        image = Image.open(imagePath).convert('RGB')
        label = np.asarray(self.data.iloc[idx, 7:55],dtype=float)
        if self.sourceTransform:
            image = self.sourceTransform(image)
        return image, label

class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, labelsFile, rootDir, imageTransform, text_hasher):
        self.data = pd.read_csv(labelsFile)
        self.rootDir = rootDir
        self.sourceTransform = imageTransform
        self.text_hasher = text_hasher

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imagePath = self.rootDir + "/" + self.data['photo_x'][idx]
        image = Image.open(imagePath).convert('RGB')
        label = np.asarray(self.data.iloc[idx, 7:55],dtype=float)
        if self.sourceTransform:
            image = self.sourceTransform(image)
        name_hash = self.text_hasher.transform([self.data['name'][idx]]).toarray()
        menu_name_hash = self.text_hasher.transform([self.data['menu_name'][idx]]).toarray()
        outlet_name_hash = self.text_hasher.transform([self.data['outlet_name'][idx]]).toarray()
        price = self.data['price'][idx]
        price = np.asarray([[price]], dtype=float)
        other = np.concatenate([name_hash, menu_name_hash, outlet_name_hash, price], axis=1)
        other = torch.from_numpy(other[0]).float()
        return image, other, label