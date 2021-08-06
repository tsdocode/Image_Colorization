import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from Dataset.ColorDataset import LabImageDataset as LID

from Model.autoencoder import AutoEncoder


os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3"

EPOCHS = 30
BATCH_SIZE = 1
LR = 0.001


#Data prepare

transforms = transforms.Compose([
        transforms.Resize(600),
])


dataset = LID("../data", transforms)
trainloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)


#Model prepare
model = AutoEncoder()

#Model to GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)



criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

#Training 

def Training():
        t = tqdm(range(EPOCHS))
        for epoch in t:  # loop over the dataset multiple times
                running_loss = 0.0
                t1 = tqdm(enumerate(trainloader, 0))
                for i, data in t1:
                        # get the inputs; data is a list of [L, ab]
                        L, ab = data

                        L = L.to(device)
                        ab = ab.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = model(L)


                        # print(outputs.shape)
                        # print(ab.shape)


                        loss = criterion(outputs, ab)
                        loss.backward()
                        optimizer.step()

                        # print statistics
                        running_loss += loss.item()
                        if i % 30 == 29:    # print every 2000 mini-batches
                                print('[%d, %5d] loss: %.3f' %
                                        (epoch + 1, i + 1, running_loss / 30))
                                running_loss = 0.0
        print('Finished Training')
        torch.save(model.state_dict(), "./model.pth")

if __name__ == "__main__":
        Training()