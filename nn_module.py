import torch
from torch import nn
# 小土堆视频中写的是Tudui(对应代码中的Latiao)
class LaTiao(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self,input):
        output=input+1
        return output
latiao=LaTiao()
x=torch.tensor(1.0)
output=latiao(x)
print(output)