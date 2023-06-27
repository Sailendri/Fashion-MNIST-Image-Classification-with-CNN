## Imports
import os
import time
import numpy as np
import torch
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy.testing as npt
#from torchsummary import summary
# from tqdm import trange

# Checks for the availability of GPU 
is_cuda = torch.cuda.is_available()
if torch.cuda.is_available():
    print("working on gpu!")
else:
    print("No gpu! only cpu ;)")
    
## The following random seeds are just for deterministic behaviour of the code and evaluation

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = '0'

############################################################################### 
#Setting Up Data Loaders

import torchvision
import torchvision.transforms as transforms
import os

if not os.path.isdir('./data'):
    os.mkdir('./data')
root = './data/'

# List of transformation on the data - here we will normalize the image data to (-1,1)
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5)),])
# Geta  handle to Load the data
training_data = torchvision.datasets.FashionMNIST(root, train=True, transform=transform,download=True)
testing_data = torchvision.datasets.FashionMNIST(root, train=False, transform=transform,download=True)

num_train = len(training_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(0.2 * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_bs = 60
test_bs = 50

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Create Data loaders which we will use to extract minibatches of data to input to the network for training
train_loader = torch.utils.data.DataLoader(training_data, batch_size=train_bs,
    sampler=train_sampler, drop_last=False)
valid_loader = torch.utils.data.DataLoader(training_data, batch_size=train_bs, 
    sampler=valid_sampler, drop_last=False)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=test_bs, 
    drop_last=False)

#Visualize Few Samples

import matplotlib.pyplot as plt
%matplotlib inline

## get a batch of data
images, labels = next(iter(train_loader))

image_dict = {0:'T-shirt/Top', 1:'Trouser', 2:'Pullover', 3:'Dress',
              4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker',
              8:'Bag', 9:'Ankle Boot'}

fig = plt.figure(figsize=(8,8))

print(images.size())
#print(labels.size())
for i in np.arange(1, 13):
    ax = fig.add_subplot(3,4,i, frameon=False)
    img = images[i][0]
    ax.set_title(image_dict[labels[i].item()])
    plt.imshow(img, cmap='gray')

#Model Building

from re import X
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels = 1,out_channels = 16,kernel_size=3, padding=1, stride=1)
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.conv_layer_2 = nn.Conv2d(in_channels = 16,out_channels = 32,kernel_size=3, padding=1, stride=1)
        self.batch_norm_2 = nn.BatchNorm2d(32)
        self.conv_layer_3 = nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size=5, padding=2, stride=1)
        self.batch_norm_3 = nn.BatchNorm2d(64)
        self.fully_conn_layer = nn.Linear(3*3*64,num_classes)
    
    
    def forward(self, x):
        # We will start with feeding the data to the first layer. 
        # We take the output x and feed it back to the next layer 
        x = self.conv_layer_1(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,kernel_size = 2, stride=2)
        x = self.conv_layer_2(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,kernel_size = 2, stride=2)
        x = self.conv_layer_3(x)
        x = self.batch_norm_3(x)
        x = F.relu(x)
        x = F.max_pool2d(x,kernel_size = 2, stride=2)
        x = self.flatten(x)
        x = self.fully_conn_layer(x)
        x = F.log_softmax(x)
        return x   

    def flatten(self, x):
        N, C, H, W = x.size()
        x = x.view(x.size(0), -1)
        return x
    
# Setting up a few learning parameters
learning_rate = 1e-2
decayRate = 0.999
epochs = 5
number_of_classes = 10

## First we will define an instance of the model to train
model = Model(num_classes=number_of_classes)
print(model)

#Moving the model to the gpu if is_cuda
if is_cuda:
  model = model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

out = torch.FloatTensor([[0.1,0.8,0.05,0.05]])
true = torch.LongTensor([1])
assert criterion(out, true), 0.8925

## training loop 

def train_model(epochs=25, validate=True):
    '''
    A function to train the model on the dataset and returns the trained model, training loss and
    validation loss for every epoch.
    
    Inputs:
        epochs: Number of times the model should be trained on the whole data.
        validate: A boolean parameter that validates on validation data.
        
    Outputs:
        model: The model trained for specified number of epochs
        training loss: A list of training losses computed for every epoch.
        validation loss: A list of validation losses computed for every epoch.
    
    '''
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
       
        training_loss = 0.0
        validation_loss = 0.0
        
        model.train()
        itr = 0
        for batch,(images,labels) in enumerate(train_loader):
            # your code here
            if is_cuda:
               labels ,images= labels.cuda(),images.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learning_rate=my_lr_scheduler.step
            training_loss += loss.item()
            
            if itr%100 == 0:
                print('Epoch %d/%d, itr = %d, Train Loss = %.3f, LR = %.3E'\
                      %(epoch, epochs, itr, loss.item(),optimizer.param_groups[0]['lr']))
            itr += 1
        train_loss.append(training_loss/len(train_loader))
        print('------------------------------------------------')
        
        if validate:
            model.eval()
            with torch.no_grad():
                itr = 0
                for batch,(images,labels)  in enumerate(valid_loader):
                    # your code here
                    if is_cuda:
                       labels ,images= labels.cuda(),images.cuda()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()
                    if itr%100 == 0:
                        print('Epoch %d/%d, itr = %d, Val Loss = %.3f, LR = %.3E'\
                              %(epoch, epochs, itr, loss.item(),optimizer.param_groups[0]['lr']))
                    itr += 1
                val_loss.append(validation_loss/len(valid_loader))
                print('################################################')
                
    return model, train_loss, val_loss
                
start = time.time()
trained_model, train_loss, val_loss = train_model(epochs, validate=True)
end = time.time()
print('Time to train in seconds ',(end - start))

# Plot the losses
it = np.arange(epochs)
plt.plot(it, train_loss, label='training loss')
plt.plot(it, val_loss, label='validation loss')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.legend(loc='upper right')
plt.show()

## Testing Loop

def evaluate_model(model, loader):
    '''
    A function to test the trained model on the dataset and print the accuracy on the testset.
    
    Inputs:
        model: Trained model
        loader: train_loader or test_loader
        
    outputs:
        accuracy. returns the accuracy of prediction
    '''
    model.eval()
    with torch.no_grad():
        correct = 0
        loss = 0
        total_samples = 0
        for images, labels in loader:
            
            if is_cuda:
               labels ,images= labels.cuda(),images.cuda()
            output = model(images)

            loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total_samples += labels.size(0)

            
        accuracy = correct/total_samples*100
        print("Total Accuracy on the Input set: {} %".format(accuracy))
        return accuracy
    
# With these settings, obtained 95% train and 91% test accuracy
tr_acc = evaluate_model(model, train_loader)
ts_acc = evaluate_model(model, test_loader)
print('Train Accuracy = %.3f'%(tr_acc))
print('Test Accuracy = %.3f'%(ts_acc))

## Visualize the test samples with predicted output and true output
images, labels = next(iter(test_loader))
# images = images.numpy()
if is_cuda:
  images = images.cuda()
  labels = labels.cuda()

out = model(images)
_, preds = torch.max(out, dim=1)

images = images.to('cpu').numpy()

fig = plt.figure(figsize=(15,15))
for i in np.arange(1, 13):
    ax = fig.add_subplot(4, 3, i)
    plt.imshow(images[i][0])
    ax.set_title("Predicted: {}/ Actual: {}".format(image_dict[preds[i].item()], image_dict[labels[i].item()]), 
                color=('green' if preds[i] == labels[i] else 'red'))