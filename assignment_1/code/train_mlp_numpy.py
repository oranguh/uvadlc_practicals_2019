"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule, LinearModule
import cifar10_utils
import matplotlib.pyplot as plt
import csv

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
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
  compare = (np.argmax(predictions, axis=1) == np.argmax(targets, axis=1))
  accuracy = np.sum(compare)/compare.size
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
  # print(FLAGS.batch_size)
  # print(FLAGS.eval_freq)
  # print(FLAGS.learning_rate)
  # print(FLAGS.max_steps)

  cifar10 = cifar10_utils.get_cifar10()

  # x, y = cifar10['train'].next_batch(BATCH_SIZE_DEFAULT)
  # x = np.reshape(x, (BATCH_SIZE_DEFAULT, -1))
  # network = MLP(x.shape[-1], dnn_hidden_units, y.shape[-1])
  network = MLP(3072, dnn_hidden_units, 10)


  criterion = CrossEntropyModule()
  plotting_accuracy = []
  plotting_loss = []

  plotting_accuracy_test = []
  plotting_loss_test = []

  for i in range(FLAGS.max_steps):
    # print("Batch number: {}".format(i))

    x, y = cifar10['train'].next_batch(FLAGS.batch_size)
    x = np.reshape(x, (FLAGS.batch_size, -1))


    output = network.forward(x)

    # loss = criterion.forward(output, y)
    # print("Average loss: {} over {} samples".format(np.mean(loss), loss.size))
    # acc = accuracy(output, y)
    dx = criterion.backward(output, y)
    network.backward(dx)

    for module in network.layers:
        if isinstance(module, LinearModule):
            # print(module.params)
            # print(asda)
            module.params['weight'] -= (module.grads['weight']) * FLAGS.learning_rate
            module.params['bias'] -= (module.grads['bias']) * FLAGS.learning_rate

    # if (i % EVAL_FREQ_DEFAULT == 0):
    if (i == FLAGS.max_steps-1):
        loss_train = criterion.forward(output, y)
        # print("Batch: {}; Average loss: {} over {} samples".format(i, np.mean(loss_train), loss_train.size))
        acc_train = accuracy(output, y)
        # print("Train accuracy is: {}%".format(acc_train))

        x_test, y_test = cifar10['test'].next_batch(5000)
        x_test = np.reshape(x_test, (5000, -1))
        output_test = network.forward(x_test)

        loss_test = criterion.forward(output_test, y_test)
        print("Average test loss: {}".format(np.mean(loss_test)))
        acc_test = accuracy(output_test, y_test)
        print("Test accuracy is: {}%".format(acc_test))

        plotting_accuracy_test.append(acc_test)
        plotting_loss_test.append(loss_test)

        plotting_accuracy.append(acc_train)
        plotting_loss.append(loss_train)

        # grid_search.append({'Learning Rate': LEARNING_RATE_DEFAULT,
        #                     'Hidden Units': dnn_hidden_units,
        #                     'Train Accuracy': acc_train,
        #                     'Train Loss': np.mean(loss_train),
        #                     'Test Accuracy': acc_test,
        #                     'Test Loss': np.mean(loss_test)})
        # print(grid_search[-1])

  # keys = grid_search[0].keys()
  # with open('grid_search.csv', 'w') as output_file:
  #   dict_writer = csv.DictWriter(output_file, keys)
  #   dict_writer.writeheader()
  #   dict_writer.writerows(grid_search)

  # plt.plot(plotting_accuracy, label='train accuracy')
  # plt.plot(plotting_accuracy_test, label='test accuracy')
  # plt.plot(plotting_loss, label='train loss')
  # plt.plot(plotting_loss_test, label='test loss')
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
