import torch
import torchvision
import numpy as np


from datetime import datetime

from models.ann_models import ANN_Model

class ToTensor(object):
    """Convert PIL Images in sample to pytorch Tensors."""

    def __call__(self, image):
        image = np.array(image, dtype=np.float32) / 255
        # numpy image: H x W
        return torch.from_numpy(image)

transform = ToTensor()

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

classes = train_set.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

model = ANN_Model(len(classes))

model = model.to(device)

learning_rate = 0.01
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def fit(model, train_loader, device, criterion, optimizer, num_epochs=1):

    total_time = 0.

    for epoch in range(num_epochs):
        train_loss = 0.
        d1 = datetime.now()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()
            train_loss += loss.item()

            average_loss = train_loss / len(train_loader)
            d2 = datetime.now()
            delta = d2 - d1
            seconds = float(delta.total_seconds())
            total_time += seconds
        print('epoch %d, train_loss: %.3f, time elapsed: %s seconds' % (epoch + 1, average_loss, seconds))
    print('total training time: %.3f seconds' % (total_time))

def test_model_accuracy(model, test_loader):
    # Calculate Accuracy         
    correct = 0.
    total = 0.
    # Iterate through test dataset
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.to('cpu') == labels).sum().item()

        accuracy = 100 * correct / total
    print('Accuracy: {}%'.format(accuracy))

fit(model, train_loader, device, criterion, optimizer)    

test_model_accuracy(model, test_loader)