### Linear Regression: EGG data -> coch eigen vector
##- Input - EEG preprocessed data, size - (64,30)
##- Output - sum of eigen vector + lambda
##- Loss - L2

import torch.nn as nn
import torch.nn.functional as F

class LNR_eigen(nn.Module):
    def __init__(self, input_size, output_size):
        super(LNR_eigen, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        
    def forward(self,x):
        x = F.normalize(x,dim=1)
        x = self.fc1(x)
        return x

