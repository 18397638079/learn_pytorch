import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset=dataset,batch_size=64)

writer=SummaryWriter("logs_maxpool")
class LaTiao(nn.Module):
    def __init__(self):
        super(LaTiao,self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=True)
    def forward(self, x):
        x=self.maxpool1(x)
        return x
step=0
latiao=LaTiao()
for data in dataloader:
    imgs,targets=data
    writer.add_images("input",imgs,step)
    output=latiao(imgs)
    writer.add_images("output",output,step)
    step+=1
writer.close()