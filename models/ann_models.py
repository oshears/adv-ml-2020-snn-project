import torch.nn as nn
import torch.nn.functional as F

class ANN_Model(nn.Module):
    def __init__(self, class_num, act=F.relu):
        super(ANN_Model, self).__init__()


        self.layer1 = nn.Linear(1 * 28 * 28, 1 * 100)
        self.act1 = act


    def forward(self, x):

        x = x.view(x.size(0), -1)

        x = self.layer1(x)
        x = self.act1(x)

        return x