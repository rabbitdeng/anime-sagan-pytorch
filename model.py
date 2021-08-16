import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from layers import selfattention,BasicBlock
import numpy as np
from functools import reduce
from layers import  ConditionalBatchNorm2d as CBN

nz = 100
nc = 3
M = 3
class Generator(nn.Module):
  def __init__(self, image_size = 64, z_dim = 100, conv_dim =64):
    super().__init__()
    repeat_num = int(np.log2(image_size)) - 3
    mult = 2 ** repeat_num

    self.l1 = nn.Sequential(
        spectral_norm(nn.ConvTranspose2d(in_channels = z_dim, out_channels = conv_dim * mult, kernel_size = 4)),
        nn.LayerNorm([512, 4, 4]),
        nn.ReLU()
    )

    curr_dim = conv_dim * mult
    self.l2 = nn.Sequential(
        spectral_norm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)),
        nn.LayerNorm([256, 8, 8]),

        nn.ReLU()
    )

    curr_dim = curr_dim // 2
    self.l3 = nn.Sequential(
        spectral_norm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)),
        nn.LayerNorm([128, 16, 16]),
        nn.ReLU()
    )

    curr_dim = curr_dim // 2
    self.l4 = nn.Sequential(
        spectral_norm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)),
        nn.LayerNorm([64, 32, 32]),
        nn.ReLU()

    )

    curr_dim = curr_dim // 2
    #self.l5 = nn.Sequential(
     #   spectral_norm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)),
     #   nn.LayerNorm([64, 64, 64]),
      #  nn.ReLU()
    #)
    self.last = nn.Sequential(
        nn.ConvTranspose2d(64, 3, 4, 2, 1),
        nn.Tanh()
        )
    self.attn1 = selfattention(128)
    self.attn2 = selfattention(64)
  def forward(self, input):
    input = input.view(input.size(0), input.size(1), 1, 1)
    out = self.l1(input) #256

    out = self.l2(out)# 128

    out = self.l3(out)# 64


    out = self.attn1(out)
    out = self.l4(out)
    out = self.attn2(out)
    out = self.last(out)
    return out



class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, image_size = 256, ndf =64):
        super().__init__()
        def conv_2d(in_channels, out_channels, kernel_size, stride = 1, padding = 0):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)),
                nn.LeakyReLU(0.1)
            )
        self.block_1 = conv_2d(in_channels, ndf, 4, 2, 1)
        current_dim = ndf
        self.block_2 = conv_2d(current_dim, current_dim * 2, 4, 2, 1)
        current_dim *= 2
        self.block_3 = conv_2d(current_dim, current_dim * 2, 4, 2, 1)
        current_dim *= 2
        #self.block_5 = conv_2d(current_dim, current_dim * 2, 4, 2, 1)
       # current_dim *= 2
        #self.block_6 = conv_2d(current_dim, current_dim * 2, 4, 2, 1)
        #current_dim *= 2
        self.attn_layer_1 = selfattention(current_dim)
        self.block_4 = conv_2d(current_dim, current_dim * 2, 4, 2, 1)
        current_dim *= 2
        self.attn_layer_2 = selfattention(current_dim)

        self.last_layer = nn.Sequential(nn.Conv2d(current_dim, 1, 4, stride= 1),
                                        )

    def forward(self, input):
        all_layers = [self.block_1, self.block_2, self.block_3, self.attn_layer_1,
                          self.block_4, self.attn_layer_2,self.last_layer]
        out = reduce(lambda x, layer: layer(x), all_layers, input)  #套娃  clock3(block2(block1(x)))......返回结果

        return out