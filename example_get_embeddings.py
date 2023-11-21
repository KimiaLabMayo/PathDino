from PathDino import get_pathDino_model
from PIL import Image
import torch

histoImg = Image.open('./inference/img.png')
model, transformInput = get_pathDino_model(weights_path='./inference/PathDino512.pth')

img = transformInput(histoImg)
embedding = model(img.unsqueeze(0))

print(embedding.shape)