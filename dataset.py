
import torch.utils.data
import numpy as np
import os, random, glob
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os 

# 数据集读取
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform

        t_dir = os.path.join(img_dir, "tampered/imgs")  #####你要训练的.jpg上一级的名字，这里设你要区分的类别为N
        u_dir = os.path.join(img_dir, "untampered")    #####
        # dir1='data\\train\\train\\tampered\\imgs'
        # dir2='data\\train\\train\\untampered'
        imgsLib = []

        # imgsLib=MyDataSet.walkFile([dir1,dir2])
        imgsLib.extend(sorted(glob.glob(os.path.join(t_dir, "*.jpg"))))   #####将图片的地址添加到列表
        imgsLib.extend(sorted(glob.glob(os.path.join(u_dir, "*.jpg"))))   #####

        # random.shuffle(imgsLib)  # 打乱数据集
        self.imgsLib = imgsLib

    # 作为迭代器必须要有的
    def __getitem__(self, index):
        img_path = self.imgsLib[index]

        label = 0 if 'untampered' in img_path.split('/') else 1                    #####为类别做标签，三类就012

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label
    
    @staticmethod
    def walkFile(file):
        tmp=[]
        for fi in file:
            for root, dirs, files in os.walk(fi):
                for f in files:
                    tmp.append(os.path.join(root, f))
        return tmp
    def __len__(self):
        return len(self.imgsLib)

#  读取数据

if __name__ == "__main__":

    CLASSES = {1: "tampered", 0: "untampered"}   ###### 标号与类别名做好对应
    img_dir = "data/train/train" #####地址为你自己train的位置，最好是绝对寻址

    data_transform = transforms.Compose([

        transforms.Resize((512,512)),  # resize到   #####规定尺寸，直到运行这部分代码可以完全看见图片
        transforms.ToTensor(),
    ])

    dataSet = MyDataSet(img_dir=img_dir, transform=data_transform)
    dataLoader = torch.utils.data.DataLoader(dataSet, batch_size=10, shuffle=True)
    for i, (data,labels) in enumerate(dataLoader):
        # get the inputs; data is a list of [inputs, labels]
        data, labels = data.to("cuda"),labels.to("cuda")
        print(labels)
    image_batch, label_batch = iter(dataLoader).next()
    for i in range(image_batch.data.shape[0]):
        label = np.array(label_batch.data[i])          
        img = np.array(image_batch.data[i]*255, np.int32)
        print(CLASSES[int(label)])
        plt.imshow(np.transpose(img, [1, 2, 0]))
        plt.show()
        plt.savefig("{}.jpg".format(i))