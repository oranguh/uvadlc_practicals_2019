"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data.
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module.

    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and
    std = 0.0001. Initialize biases self.params['bias'] with 0.

    Also, initialize gradients with zeros.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': np.random.normal(0, 0.0001, (out_features, in_features)),'bias': np.zeros(out_features)}
    self.grads = {'weight': np.zeros((out_features, in_features)),'bias': np.zeros(out_features)}
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = x @ self.params["weight"].T + self.params["bias"]
    self.x = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to
    layer parameters in self.grads['weight'] and self.grads['bias'].
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # print(dout.shape, self.x.T.shape)
    self.grads['weight'] = dout.T @ self.x
    self.grads['bias'] = dout.sum(axis=0)
    dx = dout @ self.params["weight"]
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = np.maximum(0,x)
    self.x = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    positives_x = self.x > 0
    # print(positives_x.shape)
    dx = np.multiply(positives_x, dout)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    def exp_normalize(x):
        # b = x.max()
        b = np.max(x, axis=1)
        b = b[:, np.newaxis]
        y = np.exp(x - b)
        return y / y.sum(axis = 1)[:, np.newaxis]

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = x
    self.out = exp_normalize(x)
    ########################
    # END OF YOUR CODE    #
    #######################

    return self.out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """
    def exp_normalize(x):
        b = x.max(axis=1)
        b = b[:,np.newaxis]
        b = np.repeat(b, x.shape[-1], axis=1)
        y = np.exp(x - b)
        return y / y.sum(axis=1)[:, np.newaxis]
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    expanded = exp_normalize(self.x)
    # print(expanded.shape)
    # expanded = np.expand_dims(exp_normalize(self.x.T), axis=1)
    # print(expanded.shape)
    diagonals = np.empty((expanded.shape[0], expanded.shape[1],expanded.shape[1]))

    # print(expanded[0])
    for i, batch in enumerate(expanded):
        # print(batch.size)
        diagonals[i,:,:] = np.diag(batch)
    # print(diagonals[0,:,:])
    # print(asdads)
    # print(dout.shape, self.x.shape, diagonals.shape, exp_normalize(self.x).shape)
    # dx = dout @ (np.diag(exp_normalize(self.x)) - exp_normalize(self.x) @ exp_normalize(self.x).T)

    # a = exp_normalize(self.x) @ exp_normalize(self.x).T
    a = np.einsum('ij, ik -> ijk', self.out, self.out)
    b = diagonals - a
    dx = np.einsum('ij, ijk -> ik', dout, b)

    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    small_value = 1e-8

    out = -np.log(np.sum((x + small_value)*y, axis=1))
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # print(y.shape)
    # print(np.divide(-1, x).shape)
    # print(sadasd)
    # print(x)

    # Thanks to Victor for this helpful tip
    small_value = 1e-8

    dxx = (np.divide(-1, x + small_value) * y)
    dx = dxx/y.shape[0]
    # print(dx.shape)
    # print(dx)
    # print(asda)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
