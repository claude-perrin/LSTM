#Activation Functions
#sigmoid
import numpy as np               


def sigmoid(x, derivative=False):
    """
    Computes the element-wise sigmoid activation function for an array x.

    Args:
        `x`: the array where the function is applied
        `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
    f = 1 / (1 + np.exp(-x_safe))

    if derivative:  # Return the derivative of the function evaluated at x
        return f * (1 - f)
    else:  # Return the forward pass of the function at x
        return f


def tanh_activation(x, derivative=False):
    """
    Computes the element-wise tanh activation function for an array x.

    Args:
        `x`: the array where the function is applied
        `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
    f = (np.exp(x_safe) - np.exp(-x_safe)) / (np.exp(x_safe) + np.exp(-x_safe))

    if derivative:  # Return the derivative of the function evaluated at x
        return 1 - f**2
    else:  # Return the forward pass of the function at x
        return f
#tanh activation


def softmax(x):
    """
    Computes the softmax for an array x.

    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
    f = np.exp(x_safe) / np.sum(np.exp(x_safe))

    return f
