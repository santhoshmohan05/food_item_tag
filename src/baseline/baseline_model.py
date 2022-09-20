import torch
import torch.nn as nn



class MultiModalModel(torch.nn.Module):

    def __init__(self, image_length, other_length, output_length):
        super(MultiModalModel, self).__init__()
        self.image_features = nn.Sequential(
            torch.nn.Linear(image_length, 64),
            torch.nn.ReLU(),
        )
        self.other_features = nn.Sequential(
            torch.nn.Linear(other_length, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
        )
        self.features = nn.Sequential(
            torch.nn.Linear(192, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU()
        )
        self.classifier = nn.Sequential(
            torch.nn.Linear(128, output_length),
        )

    def forward(self, image, other):
        image = self.image_features(image)
        other = self.other_features(other)
        out = torch.cat((image,other), dim=1)
        out = self.features(out)
        out = self.classifier(out)
        return out
