import torch
import torchvision.datasets
from torch import nn
from torch.nn import Sigmoid
from torch.nn import ReLU
from torch.utils.tensorboard import SummaryWriter

input=torch.tensor([[1,-0.5],
                    [-1,3]])
output=torch.reshape(input,(-1,1,2,2))
print(output.shape)

dataset=torchvision.datasets.CIFAR10(root="data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=torch.utils.data.DataLoader(dataset,batch_size=64)

class LaTao(nn.Module):
    def __init__(self):
        super(LaTao,self).__init__()
        self.relu1=ReLU()
        self.sigmoid1=Sigmoid()
    def forward(self, input):
        output=self.sigmoid1(input)
        return output
latiao=LaTao()
output=latiao(input)
print(output)

step=0
writer=SummaryWriter("logs_relu")
for data in dataloader:
    imgs,targets=data
    writer.add_images("input",imgs,step)
    output=latiao(imgs)
    writer.add_images("output",output,step)
    step=step+1
writer.close()