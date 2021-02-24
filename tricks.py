# Some tricks could make pytorch more quickly
"""
IO

- use opencv as the dataloader
- NVIDIA DALI could do data augumentation: https://zhuanlan.zhihu.com/p/77633542
- prefetch_generator would load data without stop https://zhuanlan.zhihu.com/p/97190313

CODE

- input and output channels should be same
- do not use append
- conv2d set bias = False
- .as_tensor() and .from_numpy() would not copy
- DistributedDataParallel instead of DataParallel
- set shuffle = False when eval and test
- when computational graph does not change, set torch.backends.cudnn.benchmark = True
- try to use torch.ops() rather than tensor.ops()
- transpose() rather than t() and permuter

TRAIN

- optimizer: Adamw
- mixed precision trained
"""

"""
Python
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
shell:
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

## convolution a by w
def conv(a, w):
    a = torch.randn(3, 5, 10, 10).cuda()
    w = torch.randn(7, 5, 3, 3).cuda()
    res1 = torch.nn.functional.conv2d(a, w, padding=1)
    afold = torch.nn.functional.unfold(a, 3, 1, 1)
    # batch, c, h, w -> batch, c*k*k, n. n is the feature left 
    wfold = w.view((w.size(0), -1))
    res2 = torch.matmul(wfold, afold).reshape(a.size(0), w.size(0), a.size(2), a.size(3))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
