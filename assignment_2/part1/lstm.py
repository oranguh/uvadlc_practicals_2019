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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()

        # h = hidden, x = input, p = outPut\

        self.num_hidden = num_hidden
        # torch.nn.init.normal_(tensor, mean=0, std=1)
        #  I don't understand why seq_length is a variable and also device?


        self.Wgx = nn.Parameter(nn.init.xavier_normal_((torch.empty(input_dim, num_hidden, device=device))))
        self.Wgh = nn.Parameter(nn.init.xavier_normal_((torch.empty(num_hidden, num_hidden, device=device))))
        self.Wix = nn.Parameter(nn.init.xavier_normal_((torch.empty(input_dim, num_hidden, device=device))))
        self.Wih = nn.Parameter(nn.init.xavier_normal_((torch.empty(num_hidden, num_hidden, device=device))))
        self.Wfx = nn.Parameter(nn.init.xavier_normal_((torch.empty(input_dim, num_hidden, device=device))))
        self.Wfh = nn.Parameter(nn.init.xavier_normal_((torch.empty(num_hidden, num_hidden, device=device))))
        self.Wox = nn.Parameter(nn.init.xavier_normal_((torch.empty(input_dim, num_hidden, device=device))))
        self.Woh = nn.Parameter(nn.init.xavier_normal_((torch.empty(num_hidden, num_hidden, device=device))))
        self.Wph = nn.Parameter(nn.init.xavier_normal_((torch.empty(num_hidden, num_classes, device=device))))


        self.bg = nn.Parameter(nn.init.constant_(torch.empty(num_hidden, device=device), 0))
        self.bi = nn.Parameter(nn.init.constant_(torch.empty(num_hidden, device=device), 0))
        self.bf = nn.Parameter(nn.init.constant_(torch.empty(num_hidden, device=device), 0))
        self.bo = nn.Parameter(nn.init.constant_(torch.empty(num_hidden, device=device), 0))
        self.bp = nn.Parameter(nn.init.constant_(torch.empty(num_classes, device=device), 0))

        self.device = device
        # self.myparameters = [self.Whx, self.Whh, self.Wph, self.bp, self.bh]

    def forward(self, x):

        hidden_activations = nn.init.constant_(torch.empty(x.shape[0], self.num_hidden, device=self.device), 0)
        c = nn.init.constant_(torch.empty(x.shape[0], self.num_hidden, device=self.device), 0)


        for seq_number in range(x.shape[1]):

            # print((x[:, seq_number].unsqueeze(1) @ self.Wgx).shape)

            g = torch.tanh(
                            x[:, seq_number].unsqueeze(1) @ self.Wgx +
                            hidden_activations @ self.Wgh +
                            self.bg)
            i = torch.sigmoid(
                            x[:, seq_number].unsqueeze(1) @ self.Wix +
                            hidden_activations @ self.Wih +
                            self.bi)
            f = torch.sigmoid(
                            x[:, seq_number].unsqueeze(1) @ self.Wfx +
                            hidden_activations @ self.Wfh +
                            self.bf)
            o = torch.sigmoid(
                            x[:, seq_number].unsqueeze(1) @ self.Wox +
                            hidden_activations @ self.Woh +
                            self.bo)
            c = g * i + c * f
            hidden_activations = torch.tanh(c) * o

        p = hidden_activations @ self.Wph + self.bp
        return p
