import torch
import torch.nn as nn
from torch import save , load
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as tt
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader




class Generator(nn.Module):
  def __init__(self):
    super(Generator , self).__init__()
    self.model = nn.Sequential(
        # in: batch_size x 1 x 1 x 1
        nn.ConvTranspose2d(1, 512, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        # out: batch_size x 512 x 4 x 4

        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        # out: batch_size x 256 x 8 x 8

        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        # out: batch_size x 128 x 16 x 16

        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        # out: batch_size x 64 x 32 x 32

        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        # out: batch_size x 32 x 64 x 64


        nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh()
        # out: batch_size x 1 x 128 x 128
    )

  def forward(self,x):
    return self.model(x)


gen = Generator()

with open("generator_model.pt" , "rb") as f:
    gen.load_state_dict(load(f))



## generate as much image as you want

