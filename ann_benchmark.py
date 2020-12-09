import torch
import torchvision
import numpy as np


from datetime import datetime

from models.ann_models import ANN_Model

class ToTensor(object):
    """This class is a transformer normalizing the pixels of the input image
    so that they will lie into the range of (0, 1). The input image is assumed
    to be grayscaled. The transformer is applied every time we access an image 
    through Pytorch Dataset class. For example, if we have dataset of type Dataset and
    access zeroth image like dataset[0], the image would be first transformed into an array 
    with all the pixels having a value between 0 and 1, and then the transformed image would be
    returned.
    """

    ## This method is called whenever we access an image from the dataset through dataset[i] with i as the index of the image in the dataset
    def __call__(self, image):
        image = np.array(image, dtype=np.float32) / 255
        return torch.from_numpy(image)

## instantiation of the above transformer to pass it to the dataset class
transform = ToTensor()

# Instantiating MNIST dataset. We set download as true so that the dataset class downloads the images if the images are not available on the path
# provided by root. By setting train to true, we will have only the data points associated with the training set
# of MNIST data set. Also we pass the transformer we initialized in the above line to have every image transformed while we access it.
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# The same as above with the difference that this time we want the test data
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Train loader loads the data from the data set and enumerates the data set passed to it in the way that it breaks the data down into chunks of size 32
# which is the batch size we set for training, and then it returns a permutation of these chunks every time we iterate through 
# the data set using this train loader. By setting shuffle to true, we instruct train loader to return a random permutation for each iteration.
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)


## For test data set, it doesn't matter to shuffle the data set each time, because test data set is fed into the model once
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

classes = train_set.classes

# Whether we have GPU or CPU only
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

## instantiating the model of the architecture imported from models package.
model = ANN_Model()

# moving the network into the GPU if it's available.
model = model.to(device)

learning_rate = 0.01
## Negative like likelihood loss
criterion = torch.nn.CrossEntropyLoss()
## The object running backpropagation algorithm given the parameters of the model along with the learning rate.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


## This function trains the passed model on the data set with train_loader holding reference to the data set with the loss function
## passed as well as the optimizer running back propagation for num_epochs epochs.
def fit(model, train_loader, device, criterion, optimizer, num_epochs=1):

    total_time = 0.

    for epoch in range(num_epochs):
        train_loss = 0.
        d1 = datetime.now()
        
        ## iterating the data set in a way that for each iteration, we have batch_size number of images along with their labels
        for images, labels in train_loader:
            ## we should move both images and labels to the GPU to feed them into the model which is on GPU if we have some GPU available
            images = images.to(device)
            labels = labels.to(device)

            # Clear gradients w.r.t. parameters because Pytorch stores the values by default
            optimizer.zero_grad()

            # Forward pass to get the output without softmax applied. The loss function itself applies softmax
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()
            train_loss += loss.item()

            
        ## approximating average loss of the model throughout the epoch
        average_loss = train_loss / len(train_loader)
        d2 = datetime.now()
        delta = d2 - d1
        seconds = float(delta.total_seconds())
        total_time += seconds
        print('epoch %d, train_loss: %.3f, time elapsed: %s seconds' % (epoch + 1, average_loss, seconds))
    print('total training time: %.3f seconds' % (total_time))

# testing the trained model on the test data set passed
def test_model_accuracy(model, test_loader):
    # Calculate Accuracy         
    correct = 0.
    total = 0.
    # Iterate through test dataset. We should also enter no_grad mode to prevent the model from calculating the information for backpropagation
    with torch.no_grad():
        for images, labels in test_loader:
            ## Forwarding the images into the network to get the outputs without softmax applied.
            outputs = model(images.to(device))
            ## Without applying softmax, we have some logits. The maximmum output would have the highest value in case we applied softmax. And so,
            ## The label would be the output having the maximmum value
            _, predicted = torch.max(outputs.data, 1)
            ## calculating total number of test examples.
            total += labels.size(0)
            ## Number of examples correctly annotated by the model
            correct += (predicted.to('cpu') == labels).sum().item()

        accuracy = 100 * correct / total
    print('Accuracy: {}%'.format(accuracy))

fit(model, train_loader, device, criterion, optimizer)    

test_model_accuracy(model, test_loader)
