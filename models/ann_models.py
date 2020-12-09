import torch.nn as nn
import torch.nn.functional as F


## The network we use for ANN benchmark. This network has two layers like what follows:
# layer1: input_size: 28 * 28 (The width and height of MNIST examples), output_size: 100
# layer2: input_size: 100, output_size: 10
# First layer applies Relu activation function on the linear combination of the inputs, while second layer does not apply any activation function.
# That's because pytorch negative log likelihood loss function itself applies softmax.
#
#
class ANN_Model(nn.Module):
    ## initializing the network like mentioned above.
    def __init__(self, act=F.relu):
        super(ANN_Model, self).__init__()


        self.layer1 = nn.Linear(1 * 28 * 28, 1 * 100)
        self.act = act
        
        self.layer2 = nn.Linear(100, 10)

    ## this method is called while we feed the data into the network in the forward pass.
    def forward(self, x):

        x = x.view(x.size(0), -1)

        x = self.layer1(x)
        x = self.act(x)
        
        x = self.layer2(x)

        return x
