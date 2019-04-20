"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    super(MLP, self).__init__()
    """
    Initializes MLP object.

    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP

    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    n_total = []
    n_total = n_hidden + [n_classes]

    self.layers = OrderedDict()

    if (len(n_total) > 1):
        self.layers["linear_0"] = nn.Linear(n_inputs, n_total[0])
        self.layers["relu_0"] = nn.ReLU(inplace=True)
        # self.layers["batchnorm_0"] = nn.BatchNorm1d(n_total[0])
        self.layers["dropout_0"] = nn.Dropout(0.3)


        for i in range(len(n_total)-1):
            self.layers["linear_{}".format(i+1)] = nn.Linear(n_total[i], n_total[i+1])
            if (i < (len(n_total)-2)):
                self.layers["relu_{}".format(i+1)] = nn.ReLU(inplace=True)
                # self.layers["batchnorm_{}".format(i+1)] = nn.BatchNorm1d(n_total[i+1])
                self.layers["dropout_{}".format(i+1)] = nn.Dropout(0.3)
    else:
        self.layers["linear_0"] = [nn.Linear(n_inputs, n_total[0])]

    # print(self.layers)
    self.classifier = nn.Sequential(self.layers)

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
    out = self.classifier(x)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
