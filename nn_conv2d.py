import torch
import torchvision
from PIL.Image import Image
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset=dataset,batch_size=64)

class LaTiao(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1=Conv2d(3,6,3,stride=1,padding=0)

    def forward(self,x):
        x=self.conv1(x)
        return x
latiao=LaTiao()
writer=SummaryWriter("logs")
step=0
for data in dataloader:
    imgs,targets=data
    l1 = latiao(imgs)
    writer.add_images("input",imgs,step)
    output = torch.reshape(l1, (-1, 3, 30, 30))
    writer.add_images("output",output,step)
    step+=1
writer.close()