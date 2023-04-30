#import pandas as pd
#  import matplotlib.pyplot as plt
from activation_functions import tanh_activation, sigmoid, softmax
import numpy as np
import dataset

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
        forward_pass = {"x_s": [], "z_s": [], "f_s": [], "i_s": [], "g_s": [], "C_s": [], "o_s": [], "h_s": [], "v_s": [], "output_s": []}

        # Append the initial cell and hidden state to their respective lists
        forward_pass["h_s"].append(h_prev)
        forward_pass["C_s"].append(C_prev)

        for x in inputs:

            # Concatenate input and hidden state
            z = np.row_stack((h_prev, x))
            forward_pass["z_s"].append(z)

            # Calculate forget gate
            f = sigmoid(np.dot(self.W_f, z) + self.b_f) 
            forward_pass["f_s"].append(f)

            # Calculate input gate
            i = sigmoid(np.dot(self.W_i, z) + self.b_i) 
            forward_pass["i_s"].append(i)

            # Calculate candidate
            g = tanh_activation(np.dot(self.W_g, z) + self.b_g)
            forward_pass["g_s"].append(g)

            # Calculate memory state
            C_prev = np.multiply(C_prev, f) + np.multiply(g,i)
            forward_pass["C_s"].append(C_prev)

            # Calculate output gate
            o = sigmoid(np.dot(self.W_o, z) + self.b_o)
            forward_pass["o_s"].append(o)

            # Calculate hidden state
            h_prev = o * tanh_activation(C_prev)
            forward_pass["h_s"].append(h_prev)

            # Calculate logits
            v = np.dot(self.W_v, h_prev) + self.b_v
            forward_pass["v_s"].append(v)

            # Calculate softmax
            output = softmax(v)
            forward_pass["output_s"].append(output)

        return forward_pass 


if __name__ == "__main__":
    lstm = LSTM(1,1)
    print(lstm.W_f)
    # Get first sentence in test set
    inputs, targets = test_set[1]

    # One-hot encode input and target sequence
    inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
    targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

    # Initialize hidden state as zeros
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))

    # Forward pass
    z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward(inputs_one_hot, h, c, params)

    output_sentence = [idx_to_word[np.argmax(output)] for output in outputs]
    print('Input sentence:')
    print(inputs)

    print('\nTarget sequence:')
    print(targets)

    print('\nPredicted sequence:')
    print([idx_to_word[np.argmax(output)] for output in outputs])

