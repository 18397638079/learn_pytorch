import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=torch.utils.data.DataLoader(dataset,batch_size=64,drop_last=True)
writer=SummaryWriter("logs_linear")
class LaTiao(nn.Module):
    def __init__(self):
        super(LaTiao,self).__init__()
        self.linear1=Linear(196608,10)
    def forward(self, input):
        output=self.linear1(input)
        return output
latiao=LaTiao()
step=0
for data in dataloader:
    imgs,targets=data
    # print(imgs.shape)
    # # output=torch.reshape(imgs,(1,1,1,-1))
    # output=torch.flatten(imgs)
    # print(output.shape)
    # output=latiao(output)
    # print(output.shape)
    writer.add_images("input",imgs,step)
    output1=torch.flatten(imgs)
    output1=latiao(output1)
    writer.add_scalar("output_mean",torch.mean(output1),step)
    step=step+1
writer.close()