import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from dataset import MyDataSet
from labelsmooth import LabelSmoothingLoss
from data.example.score import score_cls
from PIL import Image
import os
import ttach as tta
import numpy as np
def get_prob(output):
    return output[:,1].detach().cpu().numpy()

def write_csv(test_loader):
    with open(f"submission.csv", "w") as f:
        for idx,(data, target) in enumerate(test_loader):
            data, target = data.to("cuda:0"), target.to("cuda:0")
            res = detect(data)
            f.write(f"{idx} {res[0]}\n")
            f.flush()
            
def get_labels(test_loader):
    labels = np.zeros((test_size,2),dtype=object)
    labels[:,0] = ["{:04d}.jpg".format(i) for i in range(test_size)]
    for id,(_,label) in enumerate(testloader):
        labels[id][1] = label.item()
    return labels

def convert_GELU_to_SELU(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.GELU):
                setattr(model, child_name, nn.SELU())
        else:
                convert_GELU_to_SELU(child)

def freeze(model):
    for i, k in model.named_children():
        if isinstance(k, nn.BatchNorm2d):
            # print(k.__class__.__name__)
            k.eval()
        else:
            freeze(k)

def parm_freeze(model,stage):
    i = 0
    for param in model.parameters():
        i +=1
        if 0+stage*30<=i<(stage+1)*30:
            param.require_grad = True
        else:
            param.require_grad = False
    freeze(model)
    
def train(model,trainloader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    for i, (data,labels) in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        data, labels = data.to("cuda:0"),labels.to("cuda:0")
        # data = train_transform(data)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            print('\nTrain set:  Accuracy: {}/{} ({:.0f}%)\n'.format(correct, 100*batchsize,100. * correct / (batchsize*100)))
            running_loss = 0.0
            correct = 0

    scheduler.step()



def test(model,test_loader,max_score,labels):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        with open(f"submission.csv", "w") as f:
            for idx,(data, target) in enumerate(test_loader):
                data, target = data.to("cuda:0"), target.to("cuda:0")
                output = model(data)
                res = get_prob(F.softmax(output))
                f.write(f"{str(idx).zfill(4)}.jpg {res[0]}\n")
                f.flush()
                test_loss += F.cross_entropy(output, target,reduction = 'sum').item()
                
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # pred = torch.where(output > 0.5,1,0)
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    score = score_cls("submission.csv",labels)
    print("Test Score: {:.2f}".format(score))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc = 100. * correct / len(test_loader.dataset)
    if score > max_score:
            max_score = score
            print("save model")
            # 保存模型语句
            torch.save(model.state_dict(),save_path.format(size,max_score,test_acc))
    return max_score

CLASSES = {1: "tampered", 0: "untampered"}   ###### 标号与类别名做好对应
img_dir = "data/train/train" #####地址为你自己train的位置

size = 1024
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((size,size)),
    # transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
])
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30,expand=True,fill=1),
   
])

batchsize = 24
print("load data")
dataSet = MyDataSet(img_dir=img_dir,transform = data_transform)

train_size = int(0.95 * len(dataSet))
test_size = len(dataSet) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataSet, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,)
##get labels
labels = get_labels(testloader)

#####################################
print("Create model")
PATH = "Best_b2_model_1024_86.5.pth"
save_path = 'b0_{}_score_{}_acc_{}.pth'
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=2, bias=True)
  )
# model.head = nn.Linear(in_features=768, out_features=2, bias=True)
model = nn.parallel.DataParallel(model,device_ids=[0,1])
# model.load_state_dict(torch.load(PATH))
# model.load_state_dict({k.replace('model.',''):v for k,v in torch.load(PATH,map_location='cpu').items()})
# model.load_state_dict({'module.'+k:v for k,v in torch.load(PATH,map_location='cpu').items()})
model.to("cuda:0")


print(model)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)

#####################################

epoch = 100
max_acc = 0
max_score = 0
tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.flip_transform())
for epoch in range(epoch):  # loop over the dataset multiple times
    if epoch%10 == 0 and epoch>=20:
        parm_freeze(model,(epoch-20)//10)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
    train(model,trainloader, optimizer, epoch)
    # max_acc = test(model,testloader,max_acc,labels)
    max_score = test(tta_model,testloader,max_score,labels)


# optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
# for epoch in range(10,50):  # loop over the dataset multiple times
#     train(model,trainloader, optimizer, epoch)
#     max_acc = test(tta_model,testloader,max_acc)
print('Finished Training')
