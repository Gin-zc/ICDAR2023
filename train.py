import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler
from dataset import MyDataSet
from labelsmooth import LabelSmoothingLoss
from data.example.score import score_cls
from PIL import Image
import os
from sklearn.model_selection import KFold
import numpy as np
# import ttach as tta

class Swish(nn.Module):
	def __init(self,inplace=False):
		super(Swish,self).__init__()
		self.inplace=inplace
	def forward(self,x):
		if self.inplace:
			x.mul_(torch.sigmoid(x))
			return x
		else:
			return x*torch.sigmoid(x)

def convert_GELU_to_SELU(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.GELU):
                setattr(model, child_name, nn.SELU())
        else:
                convert_GELU_to_SELU(child)


def train(model,trainloader, device, optimizer, epoch, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    for i, (data,labels) in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        data, labels = data.to(device),labels.to(device)
        data = train_transform(data)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(data)

        loss = criterion(F.sigmoid(outputs), labels.float())
        loss.backward()
        optimizer.step()

        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            print('\nTrain set:  Accuracy: {}/{} ({:.0f}%)\n'.format(correct, 100 * batchsize, 100. * correct / (batchsize*100)))
            running_loss = 0.0
            correct = 0

    scheduler.step()
    return running_loss



def test(model,device,test_loader,max_acc):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.binary_cross_entropy(F.sigmoid(output), target.float(),reduction = 'sum').item()
            
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # pred = torch.where(output > 0.5,1,0)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.sampler)

    print('\nTest set  Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, 
        correct, len(test_loader.sampler), 
        100. * correct / len(test_loader.sampler)))
    test_acc = 100. * correct / len(test_loader.sampler)
    # if test_acc > max_acc:
    #         max_acc = test_acc
            # print("save model")
            # # 保存模型语句
            # torch.save(model.state_dict(),save_path)
    return test_acc,test_loss,max_acc

CLASSES = {1: "tampered", 0: "untampered"}   ###### 标号与类别名做好对应
img_dir = "data/train" #####地址为你自己train的位置

size = 512
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((size,size)),
    transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
])
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15,expand=True,fill=1),
   
])

batchsize = 8
dataSet = MyDataSet(img_dir=img_dir,transform = data_transform)

# train_size = int(0.8 * len(dataSet))
# test_size = len(dataSet) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataSet, [train_size, test_size])
k=10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
splits=KFold(n_splits=k,shuffle=True,random_state=42)
foldperf={}

# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
# testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False,)
#####################################
PATH = 'Best_aug_b0_model_bce.pth'
# save_path = 'Best_aug_b0_model_bce_reg.pth'
# model = models.efficientnet_b0(pretrained = models.EfficientNet_B0_Weights.IMAGENET1K_V1)
# model.classifier = nn.Sequential(
#     nn.Dropout(p=0.2, inplace=True),
#     nn.Linear(in_features=1280, out_features=1, bias=True)
#   )
# model = nn.parallel.DataParallel(model,device_ids=[0, 1])
# # model.load_state_dict(torch.load(PATH))
# model.load_state_dict({k.replace('model.',''):v for k,v in torch.load(PATH,map_location='cpu').items()})
# model.to(device)
# print(model)

#####################################
# criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()
# criterion = LabelSmoothingLoss(classes=2,smoothing=0.1)
# optimizer = optim.SGD(model.parameters(), lr=0.08, momentum=0.9,weight_decay=0.01)
# optimizer = optim.Adadelta(model.parameters(), lr=0.1)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

#####################################

num_epochs = 10
max_acc = 0
# tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.flip_transform())
# for epoch in range(epoch):  # loop over the dataSet multiple times
#     train(model,trainloader, optimizer, epoch)
#     max_acc = test(tta_model,testloader,max_acc)

# print('Finished Training')

# 训练开始
for fold, (train_idx,val_idx) in enumerate(splits.split(dataSet)):
    print('Fold {}'.format(fold + 1))

    # 读数据
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataSet, batch_size=batchsize, sampler=train_sampler)
    test_loader = DataLoader(dataSet, batch_size=batchsize, sampler=test_sampler)

    # 模型和优化器
    model = models.efficientnet_b0(pretrained = True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=1, bias=True))
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.04, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

    # 训练
    max_acc = 0
    for epoch in range(num_epochs):
        train_loss=train(model,train_loader,device,optimizer,epoch,criterion)
        test_acc, test_loss,max_acc=test(model,device,test_loader,max_acc)
        print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                       num_epochs,
                                                                                                       train_loss,
                                                                                                       test_loss,
                                                                                                       test_acc))
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        foldperf['fold{}'.format(fold+1)] = history
        if test_acc > max_acc:
            max_acc = test_acc
            print("save model")
            # 保存模型语句
            torch.save(model,'best_b0_model_fold_{}.pth'.format(fold))

# PATH = './model_data_aug.pth'2
# torch.save(model.state_dict(), PATH)

# ##test##
# # model.load_state_dict(torch.load(PATH))
# model.eval()
# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     for data,labels in validateloader:
#         data, labels = data.to(device),labels.to(device)
#         # calculate outputs by running images through the network
#         outputs = model(data)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

# trans = transforms.Compose([

#     transforms.Resize((size,size)),  
#     transforms.ToTensor(),
# ])

