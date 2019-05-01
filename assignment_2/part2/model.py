# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import torch.nn as nn
import torch

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()

        self.lstm_num_layers = lstm_num_layers
        self.lstm_num_hidden = lstm_num_hidden
        self.device = device
        self.feature_size = vocabulary_size

        self.lstm = nn.LSTM(input_size=self.feature_size,
                            hidden_size=self.lstm_num_hidden,
                            num_layers=self.lstm_num_layers)

        self.linear = nn.Linear(in_features = self.lstm_num_hidden,
                                out_features = self.feature_size)
    def forward(self, x, hidden = None):
        # sequence, batch, one-hot
        # 30, 64, 87

        # print(x.shape)
        # print(asdasd)
        lstm_out, hidden = self.lstm(x , hidden)
        # print(lstm_out.shape)
        # make sure batch is first for linear
        out = self.linear(lstm_out.transpose(0,1))
        # print(out.shape)
        # print(asdasd)
        return out, hidden
