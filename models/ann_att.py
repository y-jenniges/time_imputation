#!/usr/bin/env python
# coding: utf-8

# In[1]:

"""
Model reference: Liu, M., Wang, Y., Zhong, G., Liu, Y., Liu, X., Shi, J., et al. (2025). A deep learning approach of
artificial neural network with attention mechanism to predicting marine biogeochemistry data. Journal of Geophysical
Research: Biogeosciences, 130, e2024JG008386. https://doi.org/10.1029/2024JG008386
"""

import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F


# In[10]:


class ann_att(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ann_att, self).__init__()

        self.layer1 = nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=True)

        self.layer3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)

        self.layer2 = nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=True)

        self.activate = nn.LeakyReLU()

        self.activate2 = nn.ReLU()

        self.drop = nn.Dropout(0.1)

        self.w1 = nn.Parameter(torch.randn(in_dim, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(in_dim, hidden_dim))

        # self.soft = nn.PReLU()  #ReLU Hardsigmoid Tanh
        self.soft = nn.Tanh()

        self.b1 = nn.Parameter(torch.randn(hidden_dim))
        self.b2 = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x):
        '''
        x [batch_size N]

        out [batch_size]

        '''

        line_out = self.activate(self.layer1(x))

        att_1 = self.soft(torch.matmul(x, self.w1))

        out1 = line_out * torch.abs(att_1)

        att_2 = self.soft(torch.matmul(x, self.w2))

        out2 = self.activate(self.layer3(out1))

        out3 = out2

        out = self.layer2(out3)  # OUT3

        return out

