import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from PIL import Image
import json

class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, labelsFile, rootDir, imageTransform):
        self.data = pd.read_csv(labelsFile)

        self.rootDir = rootDir
        self.sourceTransform = imageTransform
        model_name ="cahya/bert-base-indonesian-522M"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        text = list(self.data['name'] + ' ' + self.data['menu_name'] + ' '+self.data['outlet_name'])
        self.encodings = self.tokenizer(text, truncation=True, padding=True)

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
        text_tokens = {k:torch.tensor(v[idx]) for k,v in self.encodings.items()}
        price = torch.FloatTensor([self.data['price'][idx]])
        label = torch.from_numpy(label)
        output = {'image': image,'price':price,'label':label}
        return {**output, **text_tokens}