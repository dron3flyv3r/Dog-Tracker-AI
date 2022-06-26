import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Modules):
    def __init__(self):
        super(Net, self).__init__()
        self.c1 = nn.Conv2d(1, 6, 3)
        self.c2 = nn.Conv2d(6, 16, 3)
        
        self.l1 = nn.Linear(16 * 6 * 6, 120)
        self.l2 = nn.Linear(120, 84)
        self.l3 = nn.Linear(84, 2)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), (2, 2)) # 6x6
        x = F.max_pool2d(F.relu(self.c2(x)), 2) # 3x3
        x = x.view(-1, self.num_flat_features(x)) # flatten the output of convolution layer to 1D vector of 16*6*6 elements (16*6*6 = 720) and then pass it to the linear layer
        x = F.relu(self.l1(x)) 
        x = F.relu(self.l2(x)) 
        x = self.l3(x)
        return x