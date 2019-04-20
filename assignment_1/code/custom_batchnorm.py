import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object.

    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability

    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormAutograd, self).__init__()

    self.gamma = nn.Parameter(torch.FloatTensor(torch.ones(n_neurons)), requires_grad=True)
    self.beta = nn.Parameter(torch.FloatTensor(torch.zeros(n_neurons)), requires_grad=True)
    self.n_neurons = n_neurons
    self.eps = eps
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization

    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor

    TODO:
      Check for the correctness of the shape of the input tensor.
      Implement batch normalization forward pass as given in the assignment.
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # if not (input.size()[1] == self.n_neurons):
    assert (input.size()[1] == self.n_neurons), "wrong sizes? input: {} output: {}".format(input.size(1), self.n_neurons)
        # print("wrong sizes? input: {} output: {}".format(input.size(1), self.n_neurons))

    # print("wrong sizes? input: {} output: {}".format(input.size(1), self.n_neurons))
    mean = input.mean(dim=0)
    variance = input.var(dim=0, unbiased=False)
    # print(mean, variance)
    x_norm = (input - mean) / (variance + self.eps)**0.5
    shifted = x_norm *self.gamma + self.beta

    ########################
    # END OF YOUR CODE    #
    #######################

    return shifted



######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
  pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This makes sure that the context objects
  are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization

    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shape (n_neurons)
      beta: mean bias tensor, applied per neuron, shape (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor

    TODO:
      Implement the forward pass of batch normalization
      Store constant non-tensor objects via ctx.constant=myconstant
      Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
      Intermediate results can be decided to be either recomputed in the backward pass or to be stored
      for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    ctx.eps = eps

    mean = input.mean(dim=0)
    variance = input.var(dim=0, unbiased=False)
    x_norm = (input - mean) / (variance + eps)**0.5
    out = x_norm * gamma + beta

    ctx.save_for_backward(input, x_norm, gamma, mean, variance)

    ########################
    # END OF YOUR CODE    #
    #######################

    return out


  @staticmethod
  def backward(ctx, grad_output):
    """
    Compute backward pass of the batch normalization.

    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments

    TODO:
      Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
      Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
      inputs to None. This should be decided dynamically.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    eps = ctx.eps
    # print(ctx)
    # print(ctx.saved_tensors[0].shape)
    # print(asdasd)
    # ctx.save_for_backward(input, x_norm, gamma, mean, variance)

    input = ctx.saved_tensors[0]
    x_norm = ctx.saved_tensors[1]
    gamma = ctx.saved_tensors[2]
    mean = ctx.saved_tensors[3]
    variance = ctx.saved_tensors[4]

    # print(ctx.needs_input_grad[0])
    # if ctx.needs_input_grad[0]:
    #     print("he llo")
    # print(asdasdasd)

    if ctx.needs_input_grad[1]:
        grad_gamma = (grad_output * x_norm).sum(dim=0)

    if ctx.needs_input_grad[2]:
        grad_beta = grad_output.sum(dim=0)

    if ctx.needs_input_grad[0]:
        B = input.shape[0]
        grad_input = (gamma * (1 / (variance)**0.5) / B) * (B * grad_output - x_norm * grad_gamma - grad_beta)
        # ugh
        # grad_input = (1 / (variance)**0.5) * (1. / B) * (B* (x_norm * gamma) - np.sum((x_norm * gamma), axis=0) - x_norm*np.sum(((x_norm * gamma))*x_norm, axis=0))



    # return gradients of the three tensor inputs and None for the constant eps
    return grad_input, grad_gamma, grad_beta, None



######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward is called.
  The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.

    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability

    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormManualModule, self).__init__()

    self.gamma = nn.Parameter(torch.FloatTensor(torch.ones(n_neurons)), requires_grad=True)
    self.beta = nn.Parameter(torch.FloatTensor(torch.zeros(n_neurons)), requires_grad=True)
    self.n_neurons = n_neurons
    self.eps = eps

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction

    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor

    TODO:
      Check for the correctness of the shape of the input tensor.
      Instantiate a CustomBatchNormManualFunction.
      Call it via its .apply() method.
    """

    #  I don't understand this, it feels very redundant?
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = my_bn_fct.apply(input, self.gamma, self.beta, self.eps)

    ########################
    # END OF YOUR CODE    #
    #######################

    return normalized
