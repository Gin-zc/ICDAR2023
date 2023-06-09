import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
import cv2
import random
import copy
from torch.autograd import Variable
import sys
import pickle
import math
from torchvision import transforms as T
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import random
import ttach as tta
from dataset import MyDataSet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget,BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

def denormalize_image(img_normalized, mean, std):
    """
    反标准化处理图像
    :param img_normalized: 经过标准化处理的图像
    :param mean: 均值，格式为 (R_mean, G_mean, B_mean)
    :param std: 标准差，格式为 (R_std, G_std, B_std)
    :return: 反标准化处理后的图像
    """
    img_denormalized = np.zeros_like(img_normalized)
    for i in range(3):
        img_denormalized[:,:,i] = (img_normalized[:,:,i] * std[i]) + mean[i]
    return img_denormalized

# while(1==1):
#     pass
CLASSES = {1: "tampered", 0: "untampered"}   ###### 标号与类别名做好对应
img_dir = "data/train/train" #####地址为你自己train的位置
size = 1024
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((size,size)),
    transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
])

dataSet = MyDataSet(img_dir=img_dir,transform = data_transform)
trainloader = torch.utils.data.DataLoader(dataSet, batch_size=1, shuffle=False)
image_batch, label_batch = iter(trainloader).next()
# for i in range(image_batch.data.shape[0]):
    

PATH = "Best_b0_model_1024_finetune_acc_90.8.pth"
model = models.efficientnet_b0()
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=2, bias=True)
  )
model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(PATH).items()})
model.to("cuda")
model.eval()


target_layer = [model.features[-1]]
img = Image.open("./data/train/train/untampered/0000.jpg").convert("RGB")

input_tensor = data_transform(img).unsqueeze(0)
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((size,size)),])

img = img_transform(img)
img = np.array(img,dtype=np.uint8)




cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)

targets = [ClassifierOutputTarget(1)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]

visualization = show_cam_on_image(np.transpose(img,[1,2,0]), grayscale_cam, use_rgb=True,image_weight=0.2)

plt.imshow(visualization, cmap='jet')
plt.savefig("./pic/gradcam.png")