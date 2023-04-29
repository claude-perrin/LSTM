#import pandas as pd
#import matplotlib.pyplot as plt
from activation_functions import tanh_activation, sigmoid, softmax
import numpy as np

class LSTM:
    def __init__(self, hidden_size, vocab_size):
        self.ltm = 0
        self.stm = 0

        z_size = hidden_size + vocab_size

        """
        Initializes our LSTM network.

        Args:
            `hidden_size`: the dimensions of the hidden state
            `vocab_size`: the dimensions of our vocabulary
            `z_size`: the dimensions of the concatenated input 
        """
        # Weight matrix (forget gate)

        self.W_f = np.zeros((z_size, 1))

        # Bias for forget gate
        self.b_f = np.zeros((hidden_size, 1))

        # Weight matrix (input gate)

        self.W_i = np.zeros((z_size, 1))

        # Bias for input gate
        self.b_i = np.zeros((hidden_size, 1))

        # Weight matrix (candidate)

        self.W_g = np.zeros((hidden_size, 1))

        # Bias for candidate
        self.b_g = np.zeros((hidden_size, 1))

        # Weight matrix of the output gate
        #TODO Not sure
        self.W_o = np.zeros((hidden_size, 1))

        # Bias for output gate
        self.b_o = np.zeros((hidden_size, 1))

        # Weight matrix relating the hidden-state to the output

        self.W_v = np.zeros((vocab_size, 1))

        # Bias for logits
        self.b_v = np.zeros((vocab_size, 1))

        self.W_f = self.init_orthogonal(self.W_f)
        self.W_i = self.init_orthogonal(self.W_i)
        self.W_g = self.init_orthogonal(self.W_g)
        self.W_o = self.init_orthogonal(self.W_o)
        self.W_v = self.init_orthogonal(self.W_v)

    def init_orthogonal(self, weight):
        ndim_x, ndim_y = weight.shape
        W = np.random.randn(ndim_x, ndim_y)
        u, s, v = np.linalg.svd(W)
        return u.astype('float32')

    def forward(self, inputs, h_prev, C_prev):
        """
        Arguments:
        x -- your input data at timestep "t", numpy array of shape (n_x, m).
        h_prev -- stm at timestep "t-1", numpy array of shape (n_a, m)
        C_prev -- ltm at timestep "t-1", numpy array of shape (n_a, m)
        Returns:
            z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s -- lists of size m containing the computations in each forward pass
        outputs -- prediction at timestep "t", numpy array of shape (n_v, m)
        """

        # Save a list of computations for each of the components in the LSTM
        x_s, z_s, f_s, i_s,  = [], [] ,[], []
        g_s, C_s, o_s, h_s = [], [] ,[], []
        v_s, output_s =  [], [] 

        # Append the initial cell and hidden state to their respective lists
        h_s.append(h_prev)
        C_s.append(C_prev)

        for x in inputs:

            # Concatenate input and hidden state
            z = np.row_stack((h_prev, x))
            z_s.append(z)

            # Calculate forget gate
            # YOUR CODE HERE!
            f = sigmoid(np.dot(self.W_f, z) + self.b_f) 
            f_s.append(f)

            # Calculate input gate
            # YOUR CODE HERE!
            i = sigmoid(np.dot(self.W_i, z) + self.b_i) 
            i_s.append(i)

            # Calculate candidate
            g = tanh_activation(np.dot(self.W_g, z) + self.b_g)
            g_s.append(g)

            # Calculate memory state
            # YOUR CODE HERE!
            C_prev = np.multiply(C_prev, f) + np.multiply(g,i)
            C_s.append(C_prev)

            # Calculate output gate
            # YOUR CODE HERE!
            o = sigmoid(np.dot(self.W_o, z) + self.b_o)
            o_s.append(o)

            # Calculate hidden state
            h_prev = o * tanh_activation(C_prev)
            h_s.append(h_prev)

            # Calculate logits
            v = np.dot(self.W_v, h_prev) + self.b_v
            v_s.append(v)

            # Calculate softmax
            output = softmax(v)
            output_s.append(output)

        return z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, output_s


if __name__ == "__main__":
    lstm = LSTM(1,1)
    print(lstm.W_f)
