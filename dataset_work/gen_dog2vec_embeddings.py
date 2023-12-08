import torch
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np

import json

from PIL import Image

import os

NORMALIZE = False
DATASET = "../medium_dataset"

class Encoder(nn.Module):
    def __init__(self, image_size, channels, embedding_dim):
        super(Encoder, self).__init__()
        # define convolutional layers
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # variable to store the shape of the output tensor before flattening
        # the features, it will be used in decoders input while reconstructing
        self.shape_before_flattening = None
        # compute the flattened size after convolutions
        flattened_size = (image_size // 8) * (image_size // 8) * 128
        # define fully connected layer to create embeddings
        self.fc = nn.Linear(flattened_size, embedding_dim)
    def forward(self, x):
        # apply ReLU activations after each convolutional layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # store the shape before flattening
        self.shape_before_flattening = x.shape[1:]
        # flatten the tensor
        x = x.view(x.size(0), -1)
        # apply fully connected layer to generate embeddings
        x = self.fc(x)
        return x

model = Encoder(
    channels=3,
    image_size=224,
    embedding_dim=1000,
    )
model.load_state_dict(torch.load("../encoder_total_dataset_100epoch_medium_train_valid", map_location=torch.device('cpu')))
model.eval()

# This transform is specific to resnet18
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

embeddings = {}
for group in ["train", "test", "valid"]:
    directory = f"{DATASET}/{group}/Images"
    with torch.no_grad():
        model.eval()
        for breed in os.listdir(directory):
            curPath = os.path.join(directory, breed)
            if os.path.isfile(curPath):
                continue
            for image in os.listdir(curPath):
                if image == ".DS_Store":
                    continue
                image_path = os.path.join(curPath, image)
                in_img = Image.open(image_path)
                pros_img = transform(in_img)
                input_batch = torch.unsqueeze(pros_img, 0)
                output = model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0) # normalized scores
                if NORMALIZE:
                    embeddings[image[:-4]] = probabilities.tolist()
                else:
                    embeddings[image[:-4]] = output.tolist()
            print(breed)
with open('dog2vec_best_embeddings.json', 'w') as f:
    json.dump(embeddings, f)
    