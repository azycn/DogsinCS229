import torch
import torchvision.models as models
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os

import pickle

from torchvision import transforms



model = models.resnet18(pretrained=True)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for group in ["train", "test", "valid"]:
    directory = f"small_stanforddogdataset/{group}_images"
    with torch.no_grad():
        model.eval()
        first = True
        index_map = {}
        idx = 0
        for breed in os.listdir(directory):
            curPath = os.path.join(directory, breed)
            if not os.path.isfile(curPath):
                for image in os.listdir(curPath):
                    if image == ".DS_Store":
                        continue
                    image_path = os.path.join(curPath, image)
                    in_img = Image.open(image_path)
                    pros_img = transform(in_img)
                    input_batch = torch.unsqueeze(pros_img, 0)
                    output = model(input_batch)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    # torch.save(probabilities, f"small_stanforddogdataset/{group}_embeddings/{image[:-4]}")
                    if first:
                        A = output.numpy()
                        first = False
                    else:
                        A = np.vstack([A, output.numpy()])
                    index_map[f"{image}"] = idx
                    idx += 1
                    print(group)
                    print(np.shape(A))
        with open(f"{group}_dic.pkl", 'wb') as fi:
            pickle.dump(index_map, fi)
        np.savetxt(f"{group}_embeddings.csv", A, delimiter=',')