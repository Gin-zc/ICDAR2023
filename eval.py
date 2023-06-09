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


PATH = "b0_1024_score_75.72254335260116_acc_84.09638554216868.pth"
model = models.efficientnet_b0()
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=2, bias=True)
  )
# model.head = nn.Linear(in_features=768, out_features=2, bias=True)

model.load_state_dict({k.replace('model.module.',''):v for k,v in torch.load(PATH).items()})
model.to("cuda")
model.eval()
tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.flip_transform())

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((1024,1024)),
    # transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
])

def detect(img_path):
    img = Image.open(img_path).convert("RGB")
    img = trans(img)
    img = img.unsqueeze(0).float()

    img = img.cuda()

    img = img.to(torch.float32)

    res = tta_model(img)
    prob = torch.nn.functional.softmax(res)
    return prob[:,1].detach().cpu().numpy()
with open(f"submission0.csv", "w") as f:
    imgfold = './data/test'    #原图
    img_dir_all = os.listdir(imgfold)
    img_dir_all = sorted(img_dir_all)
    for img_name in img_dir_all:
        img_dir = os.path.join(imgfold,img_name)
        for img_name in sorted(os.listdir(img_dir)):
            img_path = os.path.join(img_dir,img_name)
            res = detect(img_path)
            f.write(f"{img_name} {res[0]}\n")
            f.flush()
