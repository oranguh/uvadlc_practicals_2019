################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()

        # h = hidden, x = input, p = outPut\

        self.num_hidden = num_hidden
        # torch.nn.init.normal_(tensor, mean=0, std=1)
        #  I don't understand why seq_length is a variable and also device?

        self.Whx = nn.Parameter(nn.init.normal_((torch.empty(input_dim, num_hidden, device=device)), std=0.001))
        self.Whh = nn.Parameter(nn.init.normal_((torch.empty(num_hidden, num_hidden, device=device)), std=0.001))
        self.Wph = nn.Parameter(nn.init.normal_((torch.empty(num_hidden, num_classes, device=device)), std=0.001))

        self.bp = nn.Parameter(nn.init.constant_(torch.empty(num_classes, device=device), 0))
        self.bh = nn.Parameter(nn.init.constant_(torch.empty(num_hidden, device=device), 0))

        self.device = device
        # self.myparameters = [self.Whx, self.Whh, self.Wph, self.bp, self.bh]

    def forward(self, x):
        # Implementation here ...

        # set the 1st hidden activation thing to 0s
        hidden_activations = nn.init.constant_(torch.empty(x.shape[0], self.num_hidden), 0)
        hidden_activations = hidden_activations.to(self.device)


        for seq_number in range(x.shape[1]):

            hidden_activations = torch.tanh(
                                        x[:, seq_number].unsqueeze(1) @ self.Whx +
                                        hidden_activations @ self.Whh +
                                        self.bh)

        # print(hidden_activations.shape, self.Wph.shape, self.bp.shape)
        p = hidden_activations @ self.Wph + self.bp
        return p
