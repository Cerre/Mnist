import pandas as pd # to read csv and handle dataframe

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from resnet import ResNet, block, ResNet50

# Read data
df = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
# X_test = df_test.values

y = df['label'].values
X = df.drop(['label'], 1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)


BATCH_SIZE = 32

torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)

# # create feature and targets tensor for test set.
torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)

# # Pytorch train and test sets
# train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
# test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)

# # data loader
# train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
# test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784,250)
        self.linear2 = nn.Linear(250,250)
        self.linear3 = nn.Linear(250,100)
        self.linear4 = nn.Linear(100,10)
    
    def forward(self, X):
        X = self.linear1(X)
        X = F.relu(X)
        X = self.linear2(X)
        X = F.relu(X)
        X = self.linear3(X)
        X = F.relu(X)
        X = self.linear4(X)
        return F.log_softmax(X, dim= 1)



mlp = MLP()

def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())#,lr=0.001, betas=(0.9,0.999))
    error = nn.CrossEntropyLoss()
    EPOCHS = 1
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == var_y_batch).sum()
            if batch_idx % 50 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(train_loader), 
                    loss.item(), float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))


# fit(mlp, train_loader)

def evaluate(model):
#model = mlp
    correct = 0 
    for test_imgs, test_labels in test_loader:
        #print(test_imgs.shape)
        test_imgs = Variable(test_imgs).float()
        output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{:.3f}% ".format( float(correct) / (len(test_loader)*BATCH_SIZE)*100))
# evaluate(mlp)


# Using CNN's

torch_X_train = torch_X_train.view(-1, 1,28,28).float()
torch_X_test = torch_X_test.view(-1,1,28,28).float()
print(torch_X_train.shape)
print(torch_X_test.shape)
# # Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)

# # data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 5) # relu(conv(28*28*1)) = 24*24*32
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 5) # relu(max_pool(conv(24*24*32))) = 10*10*64
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 5) # relu(maxpool(conv(10*10*64))) = 3*3*64
        self.fc1 = nn.Linear(3*3*64,256)
        self.fc2 = nn.Linear(256,10)

        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 3*3*64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# net = ResNet50(1,10)
# net = 
net = mlp
it = iter(train_loader)
fit(net,train_loader)




        

