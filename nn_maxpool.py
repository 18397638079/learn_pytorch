import torch
from torch import nn
from torch.nn import MaxPool2d

input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]])
input=torch.reshape(input,(-1,1,5,5))
print(input.shape)

class LaTiao(nn.Module):
    def __init__(self):
        super(LaTiao,self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=False)
    def forward(self, input):
        output=self.maxpool1(input)
        return output
latiao=LaTiao()
output=latiao(input)
print(output)