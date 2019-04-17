"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    super(ConvNet, self).__init__()
    """
    Initializes ConvNet object.

    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem


    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.layers = OrderedDict()

    self.layers["conv_1"] = nn.Conv2d(n_channels, 64, 3, stride=1, padding=1)
    self.layers["relu_1"] = nn.ReLU(inplace=True)
    self.layers["batchnorm_1"] = nn.BatchNorm2d(64)
    self.layers["maxpool_1"] = nn.MaxPool2d(3, stride=2, padding=1)

    self.layers["conv_2"] = nn.Conv2d(64, 128, 3, stride=1, padding=1)
    self.layers["relu_2"] = nn.ReLU(inplace=True)
    self.layers["batchnorm_2"] = nn.BatchNorm2d(128)
    self.layers["maxpool_2"] = nn.MaxPool2d(3, stride=2, padding=1)

    self.layers["conv_3a"] = nn.Conv2d(128, 256, 3, stride=1, padding=1)
    self.layers["relu_3a"] = nn.ReLU(inplace=True)
    self.layers["batchnorm_3a"] = nn.BatchNorm2d(256)
    self.layers["conv_3b"] = nn.Conv2d(256, 256, 3, stride=1, padding=1)
    self.layers["relu_3b"] = nn.ReLU(inplace=True)
    self.layers["batchnorm_3b"] = nn.BatchNorm2d(256)
    self.layers["maxpool_3"] = nn.MaxPool2d(3, stride=2, padding=1)

    self.layers["conv_4a"] = nn.Conv2d(256, 512, 3, stride=1, padding=1)
    self.layers["relu_4a"] = nn.ReLU(inplace=True)
    self.layers["batchnorm_4a"] = nn.BatchNorm2d(512)
    self.layers["conv_4b"] = nn.Conv2d(512, 512, 3, stride=1, padding=1)
    self.layers["relu_4b"] = nn.ReLU(inplace=True)
    self.layers["batchnorm_4b"] = nn.BatchNorm2d(512)
    self.layers["maxpool_4"] = nn.MaxPool2d(3, stride=2, padding=1)

    self.layers["conv_5a"] = nn.Conv2d(512, 512, 3, stride=1, padding=1)
    self.layers["relu_5a"] = nn.ReLU(inplace=True)
    self.layers["batchnorm_5a"] = nn.BatchNorm2d(512)
    self.layers["conv_5b"] = nn.Conv2d(512, 512, 3, stride=1, padding=1)
    self.layers["relu_5b"] = nn.ReLU(inplace=True)
    self.layers["batchnorm_5b"] = nn.BatchNorm2d(512)
    self.layers["maxpool_5"] = nn.MaxPool2d(3, stride=2, padding=1)
    self.layers["avgpool_5"] = nn.AvgPool2d(1, stride=1, padding=0)

    self.convoluter = nn.Sequential(self.layers)

    self.classifier = nn.Sequential(nn.Linear(512, n_classes))
    print("Created model : {}".format(self))
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through
    several layer transformations.

    Args:
      x: input to the network
    Returns:
      out: outputs of the network

    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = self.convoluter(x)

    # desired = (BATCHSIZE, 512)
    out = out.view(x.shape[0], -1)

    out = self.classifier(out)


    ########################
    # END OF YOUR CODE    #
    #######################

    return out
