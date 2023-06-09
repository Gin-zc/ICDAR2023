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


def convert_GELU_to_SELU(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.GELU):
                setattr(model, child_name, nn.SELU())
        else:
                convert_GELU_to_SELU(child)

def train(model,trainloader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    for i, (data,labels) in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        data, labels = data.to("cuda:0"),labels.to("cuda:0")
        data = train_transform(data)
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



def test(model,test_loader,max_acc):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to("cuda:0"), target.to("cuda:0")
            output = model(data)

            test_loss += F.cross_entropy(output, target,reduction = 'sum').item()
            
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # pred = torch.where(output > 0.5,1,0)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc = 100. * correct / len(test_loader.dataset)
    if test_acc > max_acc:
            max_acc = test_acc
            print("save model")
            # 保存模型语句
            torch.save(model.state_dict(),save_path.format(size,max_acc))
    return max_acc

CLASSES = {1: "tampered", 0: "untampered"}   ###### 标号与类别名做好对应
img_dir = "data/train/train" #####地址为你自己train的位置

size = 1024
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

batchsize = 16
dataSet = MyDataSet(img_dir=img_dir,transform = data_transform)

train_size = int(0.9 * len(dataSet))
test_size = len(dataSet) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataSet, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False,)
#####################################
PATH = 'Best_b0_model_{}.pth'.format(size)
save_path = 'Best_b0_model_{}_acc_{}.pth'
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=2, bias=True)
  )
# model.classifier[2] = nn.Linear(in_features=768, out_features=2, bias=True)
model = nn.parallel.DataParallel(model,device_ids=[0, 1])
# model.load_state_dict(torch.load(PATH))
# model.load_state_dict({k.replace('model.',''):v for k,v in torch.load(PATH,map_location='cpu').items()})
model.to("cuda:0")
# convert_GELU_to_SELU(model)
print(model)

#####################################
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
# criterion = LabelSmoothingLoss(classes=2,smoothing=0.1)\

optimizer = optim.SGD(model.parameters(), lr=0.05,momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

#####################################

epoch = 50
max_acc = 0
tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.flip_transform())
for epoch in range(10):  # loop over the dataset multiple times
    train(model,trainloader, optimizer, epoch)
    max_acc = test(tta_model,testloader,max_acc)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
for epoch in range(10,50):  # loop over the dataset multiple times
    train(model,trainloader, optimizer, epoch)
    max_acc = test(tta_model,testloader,max_acc)
print('Finished Training')
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
#         data, labels = data.to("cuda"),labels.to("cuda")
#         # calculate outputs by running images through the network
#         outputs = model(data)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

# trans = transforms.Compose([

#     transforms.Resize((512,512)),  
#     transforms.ToTensor(),
# ])

# def detect(img_path):
#     img = Image.open(img_path).convert("RGB")
#     img = trans(img)
#     img = img.unsqueeze(0).float()

#     img = img.cuda()

#     img = img.to(torch.float32)

#     res = model(img)
#     prob = torch.nn.functional.softmax(res)
#     return prob[:,1].detach().cpu().numpy()
# with open(f"submission.csv", "w") as f:
#     imgfold = './data/test/'    #原图
#     img_dir_all = os.listdir(imgfold)
#     for img_name in img_dir_all:
#         img_dir = os.path.join(imgfold,img_name)
#         for img_name in os.listdir(img_dir):
#             img_path = os.path.join(img_dir,img_name)
#             res = detect(img_path)
#             f.write(f"{img_name} {res[0]}\n")
#             f.flush()
# labelss = [0]*500 +[1]*500
# print(labelss)
# sc =  score_cls('submission.csv',labels=labelss)
# print(sc)