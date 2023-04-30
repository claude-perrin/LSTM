#import pandas as pd
#  import matplotlib.pyplot as plt
from activation_functions import tanh_activation, sigmoid, softmax
import numpy as np
from dataset import Dataset

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

        self.W_g = np.zeros((z_size, 1))

        # Bias for candidate
        self.b_g = np.zeros((hidden_size, 1))

        # Weight matrix of the output gate
        #TODO Not sure
        self.W_o = np.zeros((z_size, 1))

        # Bias for output gate
        self.b_o = np.zeros((hidden_size, 1))

        # Weight matrix relating the hidden-state to the output

        self.W_v = np.zeros((hidden_size, 1))

        # Bias for logits
        self.b_v = np.zeros((vocab_size, 1))

        self.W_f = self.__init_orthogonal(self.W_f)
        self.W_i = self.__init_orthogonal(self.W_i)
        self.W_g = self.__init_orthogonal(self.W_g)
        self.W_o = self.__init_orthogonal(self.W_o)
        self.W_v = self.__init_orthogonal(self.W_v)

    def __init_orthogonal(self,param):
        """
        Initializes weight parameters orthogonally.
        This is a common initiailization for recurrent neural networks.

        Refer to this paper for an explanation of this initialization:
            https://arxiv.org/abs/1312.6120
        """
        if param.ndim < 2:
            raise ValueError("Only parameters with 2 or more dimensions are supported.")

        rows, cols = param.shape

        new_param = np.random.randn(rows, cols)

        if rows < cols:
            new_param = new_param.T

        # Compute QR factorization
        q, r = np.linalg.qr(new_param)

        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = np.diag(r, 0)
        ph = np.sign(d)
        q *= ph

        if rows < cols:
            q = q.T

        new_param = q

        return np.squeeze(new_param)

    def __clip_gradient_norm(grads, max_norm=0.25):
        """
        Clips gradients to have a maximum norm of `max_norm`.
        This is to prevent the exploding gradients problem.
        """ 
        # Set the maximum of the norm to be of type float
        max_norm = float(max_norm)
        total_norm = 0
        # Calculate the L2 norm squared for each gradient and add them to the total norm
        for grad in grads:
            grad_norm = np.sum(np.power(grad, 2))
            total_norm += grad_norm
        total_norm = np.sqrt(total_norm)
        # Calculate clipping coeficient
        clip_coef = max_norm / (total_norm + 1e-6)
        # If the total norm is larger than the maximum allowable norm, then clip the gradient
        if clip_coef < 1:
            for grad in grads:
                grad *= clip_coef
        return grads


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
            z = np.squeeze(np.row_stack((h_prev, x)))
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
    dataset = Dataset()
    dataset.generate_dataset()
    hidden_size = 50 # Number of dimensions in the hidden state
    vocab_size  = len(dataset.word_to_idx) # Size of the vocabulary used
    lstm = LSTM(hidden_size,vocab_size)
    # Get first sentence in test set
    inputs, targets = dataset.test_set.inputs[1], dataset.test_set.targets[1]

    # One-hot encode input and target sequence
    inputs_one_hot = dataset.one_hot_encode_sequence(inputs)
    targets_one_hot = dataset.one_hot_encode_sequence(targets)

    # Initialize hidden state as zeros
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))

    # Forward pass
    forward_pass = lstm.forward(inputs_one_hot, h, c)

    output_sentence = [dataset.idx_to_word[np.argmax(output)] for output in forward_pass["output_s"]]
    print('Input sentence:')
    print(inputs)

    print('\nTarget sequence:')
    print(targets)

    print('\nPredicted sequence:')
    print([dataset.idx_to_word[np.argmax(output)] for output in forward_pass["output_s"]])

