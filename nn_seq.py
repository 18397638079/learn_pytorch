import torch
from torch import nn
from torch.nn import Conv2d, Flatten, Linear, Sequential, MaxPool2d
from torch.utils.tensorboard import SummaryWriter


class LaTiao(nn.Module):
    def __init__(self):
        super(LaTiao,self).__init__()
        self.modle1=Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self,x):
        x=self.modle1(x)
        return x
latiao=LaTiao()
# print(latiao)
input=torch.ones(64,3,32,32)
output=latiao(input)
print(output.shape)
writer=SummaryWriter("logs_seq")
writer.add_graph(latiao,input)
writer.close()