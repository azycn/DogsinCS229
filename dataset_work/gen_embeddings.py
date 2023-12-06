import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np

import json

from PIL import Image

import os

NORMALIZE = False
DATASET = "../medium_dataset"


model = models.resnet18(pretrained=True)

# This transform is specific to resnet18
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
with open('embeddings.json', 'w') as f:
    json.dump(embeddings, f)