import numpy as np


def adam_optimizer(parameters, gradients, learning_rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Adam optimization algorithm implementation.

    Arguments:
    parameters -- dictionary containing model parameters
    gradients -- dictionary containing gradients of model parameters
    learning_rate -- learning rate for the algorithm
    beta1 -- exponential decay rate for the first moment estimates (default: 0.9)
    beta2 -- exponential decay rate for the second moment estimates (default: 0.999)
    epsilon -- small constant to prevent division by zero (default: 1e-8)

    Returns:
    parameters -- updated model parameters
    """

    # Initialize the first and second moment estimates to zero
    first_moment = {}
    second_moment = {}

    # Initialize the parameters with zeros
    for param_name, param in parameters.items():
        first_moment[param_name] = np.zeros_like(param)
        second_moment[param_name] = np.zeros_like(param)

    # Perform Adam update for each parameter
    for param_name, param in parameters.items():
        # Update first moment estimate
        first_moment[param_name] = beta1 * first_moment[param_name] + (1 - beta1) * gradients[param_name]

        # Update second moment estimate
        second_moment[param_name] = beta2 * second_moment[param_name] + (1 - beta2) * np.square(gradients[param_name])

        # Bias correction
        first_moment_corrected = first_moment[param_name] / (1 - np.power(beta1, t))
        second_moment_corrected = second_moment[param_name] / (1 - np.power(beta2, t))

        # Update parameters
        parameters[param_name] -= learning_rate * first_moment_corrected / (np.sqrt(second_moment_corrected) + epsilon)

    return parameters
