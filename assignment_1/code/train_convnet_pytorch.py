"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
  Performs training and evaluation of ConvNet model.

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  # print(FLAGS.batch_size)
  # print(FLAGS.eval_freq)
  # print(FLAGS.learning_rate)
  # print(FLAGS.max_steps)

  cifar10 = cifar10_utils.get_cifar10()

  if torch.cuda.is_available():
	  # print(torch.device('cpu'), torch.device("cuda"))
	  device = torch.device("cuda")
  else:
	  device = torch.device("cpu")

  network = ConvNet(3, 10)
  network.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(network.parameters(), lr=FLAGS.learning_rate)

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

	  out = network.forward(x)
	  loss = criterion(out, y.argmax(dim=1))
	  # print("Batch: {} Loss {}".format(i, loss))
	  acc = accuracy(out, y)
	  # print("Accuracy: {}".format(acc))

	  optimizer.zero_grad()
	  loss.backward()
	  optimizer.step()

	  if (i % FLAGS.eval_freq == 0):
		  x, y = cifar10['test'].next_batch(300)
		  x = torch.from_numpy(x)
		  y = torch.from_numpy(y)
		  x = x.to(device)
		  y = y.to(device)
		  out = network.forward(x)
		  loss_test = criterion(out, y.argmax(dim=1))
		  print("TEST Batch: {} Loss {}".format(i, loss_test))
		  acc_test = accuracy(out, y)
		  print("TEST Accuracy: {}".format(acc_test))

		  plotting_accuracy_test.append(acc_test)
		  plotting_loss_test.append(loss_test.item())
		  plotting_accuracy.append(acc)
		  plotting_loss.append(loss.item())

  plt.plot(plotting_accuracy, label='train accuracy')
  plt.plot(plotting_accuracy_test, label='test accuracy')
  # plt.plot(plotting_loss, label='train loss')
  # plt.plot(plotting_loss_test, label='test loss')
  plt.legend()
  plt.show()

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
