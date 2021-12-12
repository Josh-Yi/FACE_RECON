import torchvision
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional
from torchvision import transforms
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from LeNet5 import LeNet5

josh = np.load('imgs_1.npy')
josh_train = josh[:, :, :, :150]
josh_vali = josh[:, :, :, 150:175]

empty = np.load('imgs_2.npy')
empty_train = empty[:, :, :, :150]
empty_vali = empty[:, :, :, 150:175]

yuanhao = np.load('imgs_3.npy')
yuanhao_train = yuanhao[:, :, :, :150]
yuanhao_vali = yuanhao[:, :, :, 150:175]

xinyue = np.load('imgs_4.npy')
xinyue_train = xinyue[:, :, :, :150]
xinyue_vali = xinyue[:, :, :, 150:175]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FaceDataset(Dataset):
    def __init__(self, img, target):
        self.img = img
        self.target = target

    def __getitem__(self, idx):
        img = self.img
        img = img[:, :, :, idx]
        img = torch.from_numpy(img)
        tar = self.target
        return img, tar

    def __len__(self):
        return np.size(self.img, 3)

trans = transforms.Compose([
    transforms.ToTensor(),
])

joshset_train = FaceDataset(josh_train,torch.tensor([1,0,0,0]))
emptyset_train = FaceDataset(empty_train,torch.tensor([0,1,0,0]))
yuanhaoset_train = FaceDataset(yuanhao_train,torch.tensor([0,0,1,0]))
xinyuset_train = FaceDataset(xinyue_train,torch.tensor([0,0,0,1]))

joshset_vali = FaceDataset(josh_vali,torch.tensor([1,0,0,0]))
emptyset_vali = FaceDataset(empty_vali,torch.tensor([0,1,0,0]))
yuanhaoset_vali = FaceDataset(yuanhao_vali,torch.tensor([0,0,1,0]))
xinyueset_vali = FaceDataset(xinyue_vali,torch.tensor([0,0,0,1]))

train_set = ConcatDataset([joshset_train, yuanhaoset_train, emptyset_train,xinyuset_train])
vali_set = ConcatDataset([joshset_vali, yuanhaoset_vali,emptyset_vali,xinyueset_vali])
batch_size = 10
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valiloader = DataLoader(vali_set, batch_size=batch_size, shuffle=True)

model = LeNet5(4)
model = model.double()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), 0.001,weight_decay=5e-8 )
loss_fn = nn.BCELoss()
loss_fn = loss_fn.to(device)

epoch = 50
vali_loss_min = 100
for i in range(epoch):
    print('-' * 10)
    print('Epoch {}/{}'.format(i, epoch))
    train_loss = 0
    model = model.train()
    batch_counter = 0
    total_accuracy = 0
    for data in tqdm(trainloader):
        x,y = data
        x = x.double()
        y = y.double()
        x = x.to(device)
        y = y.to(device)
        x = x.reshape(-1,3,128,128)
        output = model(x)
        optimizer.zero_grad()
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        batch_counter += 1

        accuracy = (output.argmax(1) == y.argmax(1)).sum()
        total_accuracy = accuracy + total_accuracy
    print('\n','Epoch: {} | Training loss : {}'.format(i, train_loss / batch_counter))
    print('Train Accuracy : {:.6}%'.format(100*total_accuracy/(batch_counter*batch_size)))


    vali_loss = 0
    batch_counter = 0
    total_accuracy = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(valiloader):
            x, y = data
            x = x.double()
            y = y.double()
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(-1, 3, 128, 128)
            output = model(x)
            loss = loss_fn(output, y)
            vali_loss += loss.item()
            batch_counter+=1
            accuracy = (output.argmax(1) == y.argmax(1)).sum()
            total_accuracy = accuracy + total_accuracy
    print('\n', 'Epoch: {} | Validating loss : {}'.format(i, train_loss / batch_counter))
    print('Vali Accuracy : {:.6}%'.format(100*total_accuracy/(batch_counter*batch_size)))
    if vali_loss / batch_counter < vali_loss_min:
        vali_loss_min = vali_loss / batch_counter
        torch.save(model, 'Model.pth')
        print('Model saved at loss = {}'.format(vali_loss_min))


