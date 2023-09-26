import os
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from annoy import AnnoyIndex
import numpy as np

images_folder = 'clothes'
images = os.listdir(images_folder)
weights = models.ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights)
model.fc = nn.Identity()
print(model)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()])

annoy_index = AnnoyIndex(2048, 'angular')
s = 0
for i in range(len(images)):
    try:

        image = Image.open(os.path.join(images_folder, images[i]))
        input_tensor = transform(image).unsqueeze(0)
        input_tensor = input_tensor.to(device)
        if input_tensor.size()[1] == 3:
            output_tensor = model(input_tensor)
            annoy_index.add_item(i, output_tensor[0])
            s += 1

    except:
        pass
print(s)
annoy_index.build(100)
annoy_index.save('clothes_all_50.ann')
