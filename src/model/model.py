
from turtle import forward
import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel # BertForSequenceClassification


class TextFeatureExtractor(torch.nn.Module):
    def __init__(self, num_features):
        super(TextFeatureExtractor, self).__init__()
        self.transformer_model = BertModel.from_pretrained('cahya/bert-base-indonesian-522M')
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(768, num_features), nn.ReLU())
    
    def forward(self,text_encoding):
        out = self.transformer_model(**text_encoding)
        return self.classifier(out[1])


class FoodItemTagModel(torch.nn.Module):
    def __init__(self, image_length, text_length, output_length):
        super(FoodItemTagModel, self).__init__()
        self.image_model = self.create_image_model(image_length)
        self.text_model = TextFeatureExtractor(text_length)
        self.classifier = self.create_dense_classifier(image_length+text_length+1,output_length)

    def create_dense_classifier(self, in_features, out_features):
        return nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 2048), nn.ReLU(),
                        nn.Dropout(0.25), nn.Linear(2048, 1024), nn.ReLU(),
                        nn.Dropout(0.25), nn.Linear(1024, 512), nn.ReLU(),
                        nn.Dropout(0.3), nn.Linear(512, 512), nn.ReLU(),
                        nn.Dropout(0.3), nn.Linear(512, out_features))

    def create_image_model(self, num_features):
        model_ft = models.resnet50(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_features)
        return model_ft


    def forward(self, image, text_tokens, price):
        image_features = self.image_model(image)
        text_features = self.text_model(text_tokens)
        final_features = torch.cat([image_features, text_features, price],dim=1)
        return self.classifier(final_features)
