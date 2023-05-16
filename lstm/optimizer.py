import numpy as np


class AdamOptim():
    def __init__(self, lr=0.04, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr

    def update(self, dw, db, t=1):
        # dw, db are from current minibatch
        # momentum beta 1
        # *** weights *** #
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        # *** biases *** #
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db

        # rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw ** 2)
        # *** biases *** #
        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * (db ** 2)
        # bias correction
        m_dw_corr = self.m_dw / (1 - self.beta1 ** t)
        m_db_corr = self.m_db / (1 - self.beta1 ** t)
        v_dw_corr = self.v_dw / (1 - self.beta2 ** t)
        v_db_corr = self.v_db / (1 - self.beta2 ** t)
        # how much weights and biases should be adjusted
        dw = self.lr * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))
        # db = self.lr * (m_db_corr / (np.sqrt(v_db_corr) + self.epsilon))
        return dw  # , db


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


if __name__ == '__main__':
    adam = AdamOptim()
