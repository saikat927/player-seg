import os
import random
import json

from tqdm import tqdm
import imantics
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import albumentations as A


model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

checkpoint = torch.load('last_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

preprocess_input = get_preprocessing_fn("resnet34", pretrained="imagenet")

print('model loaded')

N_IMAGES = 512
TRAIN_IMAGE_SIZE = 512
INPUT_IMAGE_SIZE = (1920, 1080)

cur_image = cv2.imread('2.png')
cur_image = cv2.cvtColor(cur_image, cv2.COLOR_BGR2RGB)
cur_image = cv2.resize(cur_image, (TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE))
cur_image = preprocess_input(cur_image)

image = torch.tensor(cur_image, dtype=torch.float)
image = image.permute(2, 0, 1)
image = image[None, :]

print(image.shape)

with torch.no_grad():   
    
    output = model(image)
    np_outputs = output.detach().cpu().numpy()

    plt.imshow(np_outputs[0][0]>=0)
    plt.title("Predict")
    plt.show()


