"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import csv

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100, 100, 100, 100'
LEARNING_RATE_DEFAULT = 2e-2
MAX_STEPS_DEFAULT = 15000
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.

  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  compare = (predictions.argmax(dim=1)) == (targets.argmax(dim=1))
  summed = compare.sum().item()
  # print(compare.size()[0])
  accuracy = summed/compare.size()[0]
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model.

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  cifar10 = cifar10_utils.get_cifar10()

  if torch.cuda.is_available():
      # print(torch.device('cpu'), torch.device("cuda"))
      device = torch.device("cuda")
  else:
      device = torch.device("cpu")

  network = MLP(3072, dnn_hidden_units, 10)
  network.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(network.parameters(), lr=FLAGS.learning_rate) #, weight_decay=1/(200*9))
  # optimizer = optim.RMSprop(network.parameters(), lr=FLAGS.learning_rate)
  # optimizer = optim.SGD(network.parameters(), lr=FLAGS.learning_rate)

  # print(FLAGS.batch_size)
  # print(FLAGS.eval_freq)
  # print(FLAGS.learning_rate)
  # print(FLAGS.max_steps)

  plotting_accuracy = []
  plotting_loss = []
  plotting_accuracy_test = []
  plotting_loss_test = []

  for i in range(1, FLAGS.max_steps-1):

      x, y = cifar10['train'].next_batch(FLAGS.batch_size)
      x = torch.from_numpy(x)
      y = torch.from_numpy(y)
      x = x.to(device)
      y = y.to(device)

      x = x.view(FLAGS.batch_size, -1)


      out = network.forward(x)
      loss = criterion(out, y.argmax(dim=1))
      # print("Batch: {} Loss {}".format(i, loss))
      # acc = accuracy(out, y)
      # print("Accuracy: {}".format(acc))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # learning_rate = 0.01
      # for f in network.parameters():
      #     f.data.sub_(f.grad.data * learning_rate)

      # if (i % FLAGS.eval_freq == 0):
      #     # print("TRAIN Batch: {} Loss {}".format(i, loss.item()))
      #     acc = accuracy(out, y)
      #     print("TRAIN Accuracy: {}".format(acc))
      #     plotting_accuracy.append(acc)
      #     plotting_loss.append(loss.item())
      #
      #     x, y = cifar10['test'].next_batch(5000)
      #     x = torch.from_numpy(x)
      #     y = torch.from_numpy(y)
      #     x = x.to(device)
      #     y = y.to(device)
      #     x = x.view(5000, -1)
      #     out = network.forward(x)
      #     loss = criterion(out, y.argmax(dim=1))
      #     # print("TEST Batch: {} Loss {}".format(i, loss))
      #     acc = accuracy(out, y)
      #     print("TEST Accuracy: {}".format(acc))
      #     # print(loss.item())
      #     # print(asdasd)
      #     plotting_accuracy_test.append(acc)
      #     plotting_loss_test.append(loss.item())

      if (i == FLAGS.max_steps-FLAGS.eval_freq):
          print("hellooo")
          acc = accuracy(out, y)
          print("TRAIN Accuracy: {}".format(acc))
          train_accuracy = acc
          train_loss = loss.item()

          x, y = cifar10['test'].next_batch(5000)
          x = torch.from_numpy(x)
          y = torch.from_numpy(y)
          x = x.to(device)
          y = y.to(device)
          x = x.view(5000, -1)
          out = network.forward(x)
          loss = criterion(out, y.argmax(dim=1))
          acc = accuracy(out, y)
          print("TEST Accuracy: {}".format(acc))

          test_accuracy = acc
          test_loss = loss.item()

          with open('MLP_results.csv', 'a') as output_file:
              writer = csv.writer(output_file)
              writer.writerow([FLAGS.dnn_hidden_units,
                                FLAGS.learning_rate,
                                train_accuracy,
                                train_loss,
                                test_accuracy,
                                test_loss])


  # plt.plot(plotting_accuracy, label='train accuracy')
  # plt.plot(plotting_accuracy_test, label='test accuracy')
  # plt.plot(plotting_loss, label='train loss')
  # plt.plot(plotting_loss_test, label='test loss')
  # plt.tight_layout()
  # plt.legend()
  # plt.show()



  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
