"""
# Multiclass Classification of Images using Convolutional Neutral Networks 

Classifying images of handwritten digits using CNNs. The dataset used is the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database).

The notebook is broken into the following sections:
1. Loading MNIST 
2. Building the CNN
3. Building the Trainer
4. Training the Network
5. Network Accuracy
6. Visualizing CNN Layers
7. Comparing with a Fully Connected Network
"""

!pip install torchviz

# Commented out IPython magic to ensure Python compatibility.
# Setting some hyperparameters and making sure we have a GPU
# %matplotlib inline
import numpy as np 
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
from torchviz import make_dot

# Set the device to use
# CUDA refers to the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Hyperparameters
num_epochs = 10
num_classes = 10  # there are 10 digits: 0 to 9
batch_size = 256

## Fixing Random Seed for Reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# If you are on CoLab and successfully using the GPU, this print should
#   contain "cuda" in it
print(str(device))
assert('cuda' in str(device))  # comment out this assert if you are not using a GPU

"""### Loading MNIST
Here we are loading the MNIST dataset. This dataset consists of 60,000 training images and 10,000 test images.

Each image is a 28-by-28 grayscale image, so it is represented by a 28-by-28 array. The labels are 0 to 9, representing digits 0-9.
"""

from torch.utils.data import DataLoader

# transforms to apply to the data, converting the data to PyTorch tensors and normalizing the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=trans)

# train_loader returns batches of training data. See how train_loader is used in the Trainer class later
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,num_workers=0)

##ASSERTS: checking whether the data is loaded as expected
assert(len(train_loader)*batch_size >= 60000)
assert(len(test_loader)*batch_size >=10000)

"""### Building the network
Let's build a CNN to classify MNIST images. We will build a CNN with the following architecture:

Input:
0. The input is a 28-by-28 image with only 1 channel (since it is grayscale)

Network:
1.   2D Convolutional Layer with 32 output channels, 5-by-5 kernels, and padding of size 2, activation function RELU
2.   Max Pooling with a 2-by-2 kernel and a stride of size 2
3.   2D Convolutional Layer with 64 output channels, 5-by-5 kernels, and padding of size 2, activation function RELU
4.   Max Pooling with a 2-by-2 kernel and a stride of size 2
5.   Fully Connected Layer with output size 512 with RELU activation
6.   Fully Connected Layer with output of size 10 (no activation function)

"""

import torch.nn as nn
import torch.nn.functional as F


### Code below based on the architecture described above
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Compute the size of the the input for the first fully connected layer.
        # We can track what happens to a 28-by-28 image when passes through the previous layers.
        # We will endup with 64 channels, each of size x-by-x, 
        # therefore the size of input is (64*x*x) 
        self.size_linear = 64*7*7
        self.fc1 = nn.Linear(self.size_linear, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x))) 
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self.size_linear) # this flattens x into a 1D vector
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
##ASSERT: checks if our CNN has the correct output shape
with torch.no_grad():  # tells PyTorch not to track gradients here
    # test_data is 100 random images, 1 channel, 28-by-28
    test_data = torch.rand(100,1,28,28)
    test_net = ConvNet()
    out = test_net.forward(test_data)
    # the output should have size (100,10)
    assert(out.size()==(100,10))

"""### Visualize Our Network

Use the `made_dot()` function from torchvis to visualize the network. Check https://github.com/szagoruyko/pytorchviz for more detail.
"""

test_data = torch.rand(100,1,28,28)
test_net = ConvNet()
out = test_net.forward(test_data)
make_dot(out)  

"""### Building the trainer
To train our model, we'll build a Trainer class that holds our network and data. When we call 

```
trainer.train(epochs)
```
The Trainer trains over all the data for *epochs* times. 
It iterates over batches of data from the train_loader, passes it through the network, computes the loss and the gradients and lets the optimizer (SGD in this case) update the parameters.

"""

from os import X_OK
class Trainer():
    def __init__(self,net=None,optim=None,loss_function=None, train_loader=None):
        self.net = net
        self.optim = optim
        self.loss_function = loss_function
        self.train_loader = train_loader

    def train(self,epochs):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            for data in self.train_loader:
                
                # Moving this batch to GPU
                # Note that X has shape (batch_size, number of channels, height, width)
                # which is equal to (256,1,28,28) since our default batch_size = 256 and 
                # the image has only 1 channel
                X = data[0].to(device)
                y = data[1].to(device)
                
                # Zero the gradient in the optimizer i.e. self.optim
                self.optim.zero_grad()

                # Getting the output of the Network
                output = self.net.forward(X)

                # Computing loss using loss function i.e. self.loss_function
                loss = self.loss_function(output, y)

                # Backpropagate to compute gradients of parameteres
                loss.backward()

                # Call the optimizer i.e. self.optim
                self.optim.step()

                epoch_loss += loss.item()
                epoch_steps += 1

            # average loss of epoch
            losses.append(epoch_loss / epoch_steps)
            print("epoch [%d]: loss %.3f" % (epoch+1, losses[-1]))

        return losses

"""### Training the network 
Let's find the right learning rate. Test out training using various learning rates (we'll want to reinitialize the network each time we choose a new learning rate).

Let's find a learning rate that results in less than 0.03 loss after 10 epochs of training. 
"""

import torch.optim as optim

### Try different learning rates for SGD to see which one works (do not try learning rates greater than 1)
### number of epochs is fixed at 10, do not change it
### we want the last epoch loss to be less than 0.03
learning_rate = 0.01

net = ConvNet()
net = net.to(device)
opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
loss_function = nn.CrossEntropyLoss()

trainer = Trainer(net=net, optim=opt, loss_function=loss_function, train_loader=train_loader)

losses = trainer.train(num_epochs)
###ASSERTS
assert(losses[-1] < 0.03)
assert(len(losses)==num_epochs)  # because we record the loss after each epoch

import matplotlib.pyplot as plt
### Plot the training loss (y-axis) vs epoch number (x-axis)
### using the losses computed in previous step

epochs = np.arange(1, num_epochs+1)
plt.plot(epochs, losses)

plt.xlabel('Epoch Number')
plt.ylabel('Training Losses')

plt.xticks(epochs)
plt.xlim([1, num_epochs])

plt.title('Training Losses versus Epoch Number')

plt.show()

"""### Accuracy of our network on test data
Compute the accuracy of the network on the test data. If the CNN is working correctly, we should get accuracy of >98% on the test data.
"""

err = 0
tot = 0
with torch.no_grad():
    for data in test_loader:
        # Retrieve X and y for this batch, from data, and 
        # move it to the device we are using (probably the GPU)
        # (like what we did in trainer)
        X = data[0].to(device)
        y = data[1].to(device)

        # raw output of network for X
        output = net(X)
        
        # let the maximum index be our predicted class
        _, yh = torch.max(output, 1) 

        # tot will 10,000 at the end, total number of test data
        tot += y.size(0)

        ## Add to err number of missclassification, i.e. number of indices that 
        ## yh and y are not equal
        ## note that y and yh are vectors of size = batch_size = (256 in our case)
        filter = (y != yh)
        ones = torch.ones(y.shape)
        err += torch.sum(ones[filter])

print('Accuracy of prediction on test digits: %5.2f%%' % (100-100 * err / tot))

###ASSERTS
assert((100-100 * err / tot)>=98)
assert(tot==10*1000)

"""### Inspecting Model Errors

**Brief description of what we observe when the model makes incorrect predictions. Is the model making obvious mistakes? Or is the data also tricky?**

The model is not making obvious mistakes. The data is tricky. 
For the three examples inspected when the model was run, the first digit could be a 4 or a 9. 
The second digit doesn't look like a digit at all, but more like a capital H. 
The third digit looks like a 6 but the top of the 6 is not fully drawn so it could be a 0.
"""

import numpy as np
import matplotlib.pyplot as plt

### function for normalizing a 2d image (input type = numpy 2d array)
def normalize_image(npimg):
    npimg = (npimg - np.mean(npimg)) / np.std(npimg)
    npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
    return npimg

num_to_check = 3 # inspect 3 examples
num_checked = 0
test_loader_for_error_analysis = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
net.eval()
with torch.no_grad():
    for data in test_loader_for_error_analysis:
        # Retrieve X and y for this batch from data and 
        # move it to GPU (what we did in trainer)
        X = data[0].to(device)
        y = data[1].to(device)

        # raw output of network for X
        output = net(X)
        
        # let the maximum index be our predicted class
        _, yh = torch.max(output, 1)
        
        # check incorrect prediction
        if yh.item() != y.item():
            plt.figure()
            npimg = X[0][0].to('cpu').numpy()
            npimg = normalize_image(npimg)
            plt.imshow(npimg,cmap="gray",vmin=0,vmax=1)
            plt.title("model predicted digit %d" % yh.item())
            plt.axis("off")
            num_checked += 1
            if num_checked == num_to_check:
                break

"""### Visualize CNN layers
Now we will visualize some internal values in the CNN. We will visualize the filters in the CNN and the result of applying those CNN filters to input images.

#### Visualize each filter separately in the first layer

* Our first layer was a 2d convolutional layer with 32 output channels and 5-by-5 kernel
* Therefore we have 32 different learnt filters. Each has size (1,5,5), so, each filter is a 5-by-5 array of weights
* Let's look at each filter as a 5-by-5 grayscale image and plot it


**Brief explanation of what are these filters are detecting.**

These filters are detecting certain features within the images. 
For example, one filter detects straight lines, another filter detects edges, another filter detects corners, and another filter detects curves.
"""

plt.figure(figsize=(10,10))
for i in range(32):
    plt.subplot(4,8,i+1)
    npimg = net.conv1.weight.cpu().detach().numpy()[i][0]
    # npimg should be a 5-by-5 numpy array corresponding to the i-th filter
    # if you need to move npimg off the GPU, you can use .cpu()
    npimg = normalize_image(npimg)
    plt.imshow(npimg,cmap="gray",vmin=0,vmax=1)
    plt.title("filter "+str(i+1))
    plt.axis("off")
plt.show()

"""### Visualize the input after applying the first layer

*  First layer has 32 filters
*  Since padding is 2 and kernel is 5-by-5, each output channel will be again 28-by-28
*  Let's visualize each of these 32 pictures for one example of each digit

**Brief explanation of what these images represent.**

These images represent the original images but with the filters above applied. 
For example, an image with a straight line filter applied, any straight lines in that image will be exacerbated or made more pronounced to produce a transformed image, like those shown below.
"""

### this code picks one sample from each label (each digit) for visualizing purposes
sample_digits = dict()
for data in train_loader:
    for i in range(data[1].shape[0]):
        if data[1][i].item() not in sample_digits.keys():
            sample_digits[data[1][i].item()]=data[0][i]
    if len(sample_digits.keys())==10:
        break

for digit in range(10):
    plt.figure()
    data = sample_digits[digit]
    npimg = data[0].numpy()
    npimg = normalize_image(npimg)
    plt.imshow(npimg,cmap="gray",vmin=0,vmax=1)
    plt.title("original image of digit %d"%digit)
    plt.axis("off")
    plt.figure(figsize=(20,20))
    
    with torch.no_grad():
        data = data.unsqueeze(0).to(device)
        ### data has shape (1,1,28,28)
        ### pass the data to only layer conv1 and apply RELU activation (do not apply maxpooling)
        ### the output should be a tensor of size (1,32,28,28)
        output = F.relu(net.conv1(data))

    
    data_numpy = output.detach().cpu().numpy()
    for i in range(32):
        plt.subplot(4,8,i+1)
        npimg = data_numpy[0,i]
        npimg = normalize_image(npimg)
        plt.imshow(npimg,cmap="gray",vmin=0,vmax=1)
        plt.title("output of filter "+str(i+1))
        plt.axis("off")
    plt.show()
    
    ###ASSERTS
    assert(data.size()==(1,1,28,28))
    assert(output.size()==(1,32,28,28))

"""### Comparing with a Fully Connected Neural Network

Let's make a fully connected (FC) neural network with a similar number of parameters as our ConvNet. Then, we can compare the performance between the FC net and the ConvNet.

First, let's count the parameters in the ConvNet. The first convolutional layer has 32 5x5 filters, so 32\*5\*5 = 800 parameters.

The second convolutional layer has 64 5x5 filters, so 64\*5\*5 = 1600 parameters.

The first fully connected layer has an input of size 64\*x\*x = 64x^2. We already calculated x earlier, but here let's just estimate x as approximately 10, so we'll say the fully connected layer has an input of size 6400, and the output size is 512. So the first fully connected layer has around 6400\*512 = 3276800 parameters.

The last layer has input size 512 and output size 10, so it has 5120 parameters.

In total that is roughly 3276800 + 5120 + 800 + 1600 = 3284320 parameters.

----------------------------------------------------------

For a fully connected network, the input will have size 28 \* 28 = 784 and the output will have size 10. Let's use 3 hidden layers, with sizes 1500, 1000 and 500. Then the total number of parameters will be:

784\*1500 + 1500\*1000 + 1000\*500 + 500\*10 = 3181000 parameters, which is very similar to our ConvNet. Now let's compare the performance!

**Brief explanation why Fully Connected Layer performs better or worse than CNN.**

The Fully Connected Layer performs worse than CNN because using a Fully Connected Layer produced more misclassifications. 
The Fully Connected Layer uses much more parameters than CNN, making it susceptible to overfitting the data. 
In addition, the Fully Connected Layer takes longer to compute due to its large number of parameters.
"""

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.input_size = 28 * 28
        self.fc1 = nn.Linear(self.input_size, 1500)
        self.fc2 = nn.Linear(1500, 1000)
        self.fc3 = nn.Linear(1000, 500)
        self.fc4 = nn.Linear(500, 10)
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

learning_rate = 0.01

net = FCNet()
a = net.to(device)
opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
loss_function = nn.CrossEntropyLoss()

trainer = Trainer(net=net,optim=opt,loss_function=loss_function, train_loader=train_loader)

losses = trainer.train(num_epochs)

plt.plot(losses, linewidth=2, linestyle='-', marker='o')

err = 0
tot = 0
net.eval()
with torch.no_grad():
    for data in test_loader:
        # Retrieve X and y for this batch from data and 
        # move it to GPU 
        X = data[0].to(device)
        y = data[1].to(device)

        # raw output of network for X
        output = net(X)
        
        # let the maximum index be our predicted class
        _, yh = torch.max(output, 1) 

        # tot will 10,000 at the end, total number of test data
        tot += y.size(0)

        ## add to err number of missclassification, i.e. number of indices that 
        ## yh and y are not equal
        ## note that y and yh are vectors of size = batch_size = (256 in our case)
        filter = (y != yh)
        ones = torch.ones(y.shape)
        err += torch.sum(ones[filter])

print('Accuracy of FC prediction on test digits: %5.2f%%' % (100-100 * err / tot))