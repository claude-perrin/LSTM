# import pandas as pd
#  import matplotlib.pyplot as plt
from activation_functions import tanh_activation, sigmoid, softmax
import numpy as np
from dataset import Dataset
from sklearn.metrics import mean_squared_error


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
        w_f = np.zeros((hidden_size, z_size))
        # Bias for forget gate
        self.b_f = np.zeros((hidden_size, 1))

        # Weight matrix (input gate)

        w_i = np.zeros((hidden_size, z_size))

        # Bias for input gate
        self.b_i = np.zeros((hidden_size, 1))

        # Weight matrix (candidate)

        w_g = np.zeros((hidden_size, z_size))

        # Bias for candidate
        self.b_g = np.zeros((hidden_size, 1))

        # Weight matrix of the output gate
        # TODO Not sure
        w_o = np.zeros((hidden_size, z_size))

        # Bias for output gate
        self.b_o = np.zeros((hidden_size, 1))

        # Weight matrix relating the hidden-state to the output

        w_v = np.zeros((vocab_size, hidden_size))

        # Bias for logits
        self.b_v = np.zeros((vocab_size, 1))

        self.W_f = self.__init_orthogonal(w_f)
        self.W_i = self.__init_orthogonal(w_i)
        self.W_g = self.__init_orthogonal(w_g)
        self.W_o = self.__init_orthogonal(w_o)
        self.W_v = self.__init_orthogonal(w_v)

        print('W_i:', self.W_i.ndim)
        print('W_g:', self.W_g.ndim)
        print('W_o:', self.W_o.ndim)
        print('W_v:', self.W_v.ndim)
        print('b_i:', self.b_i.ndim)
        print('b_g:', self.b_g.ndim)
        print('W_f:', self.W_f.ndim)
        print('b_o:', self.b_o.ndim)
        print('b_v:', self.b_v.ndim)

    def __init_orthogonal(self, param):
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

        return new_param

    def __clip_gradient_norm(self, grads, max_norm=0.25):
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
        forward_pass = {"z_s": [], "f_s": [], "i_s": [], "g_s": [], "C_s": [], "o_s": [], "h_s": [], "v_s": [],
                        "output_s": []}

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
            C_prev = np.multiply(C_prev, f) + np.multiply(g, i)
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

    def __cross_entropy(self, predictions, targets, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.
        Input: predictions (N, k) ndarray
               targets (N, k) ndarray
        Returns: scalar
        """
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
        return ce
    def backward(self, forward_pass, targets, h_prev, c_prev):
        """
        Arguments:
        z_s -- your concatenated input data  as a list of size m.
        f_s -- your forget gate computations as a list of size m.
        i_s -- your input gate computations as a list of size m.
        g_s -- your candidate computations as a list of size m.
        C_s -- your Cell states as a list of size m+1.
        o_s -- your output gate computations as a list of size m.
        h_s -- your Hidden state computations as a list of size m+1.
        v_s -- your logit computations as a list of size m.
        outputs -- your outputs as a list of size m.
        targets -- your targets as a list of size m.
        W_f -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
        b_f -- Bias of the forget gate, numpy array of shape (n_a, 1)
        W_i -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
        b_i -- Bias of the update gate, numpy array of shape (n_a, 1)
        W_g -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
        b_g --  Bias of the first "tanh", numpy array of shape (n_a, 1)
        W_o -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
        b_o --  Bias of the output gate, numpy array of shape (n_a, 1)
        W_v -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_v, n_a)
        b_v -- Bias relating the hidden-state to the output, numpy array of shape (n_v, 1)
        Returns:
        loss -- crossentropy loss for all elements in output
        grads -- lists of gradients of every element in p
        """

        # Unpack parameters
        # Initialize gradients as zero
        W_f_d = np.zeros_like(self.W_f)
        b_f_d = np.zeros_like(self.b_f)

        W_i_d = np.zeros_like(self.W_i)
        b_i_d = np.zeros_like(self.b_i)

        W_g_d = np.zeros_like(self.W_g)
        b_g_d = np.zeros_like(self.b_g)

        W_o_d = np.zeros_like(self.W_o)
        b_o_d = np.zeros_like(self.b_o)

        W_v_d = np.zeros_like(self.W_v)
        b_v_d = np.zeros_like(self.b_v)

        # Set the next cell and hidden state equal to zero
        dh_next = np.zeros_like(forward_pass["h_s"][0])
        dC_next = np.zeros_like(forward_pass["C_s"][0])

        # Track loss
        loss = 0

        for t in reversed(range(len(forward_pass["output_s"]))):
            # Compute the cross entropy
            loss += mean_squared_error(forward_pass["output_s"][t], targets[t])
            # Get the previous hidden cell state
            C_prev = forward_pass["C_s"][t - 1]

            # Compute the derivative of the relation of the hidden-state to the output gate
            dv = np.copy(forward_pass["output_s"][t])
            dv[np.argmax(targets[t])] -= 1

            # Update the gradient of the relation of the hidden-state to the output gate
            W_v_d += np.dot(dv, forward_pass["h_s"][t].T)
            b_v_d += dv

            # Compute the derivative of the hidden state and output gate
            dh = np.dot(self.W_v.T, dv)
            dh += dh_next
            do = dh * tanh_activation(forward_pass["C_s"][t])
            do = sigmoid(forward_pass["o_s"][t], derivative=True) * do

            # Update the gradients with respect to the output gate
            W_o_d += np.dot(do, forward_pass["z_s"][t].T)
            b_o_d += do

            # Compute the derivative of the cell state and candidate g
            dC = np.copy(dC_next)
            dC += dh * forward_pass["o_s"][t] * tanh_activation(tanh_activation(forward_pass["C_s"][t]),
                                                                derivative=True)
            dg = dC * forward_pass["i_s"][t]
            dg = tanh_activation(forward_pass["g_s"][t], derivative=True) * dg

            # Update the gradients with respect to the candidate
            W_g_d += np.dot(dg, forward_pass["z_s"][t].T)
            b_g_d += dg

            # Compute the derivative of the input gate and update its gradients
            di = dC * forward_pass["g_s"][t]
            di = sigmoid(forward_pass["i_s"][t], True) * di
            W_i_d += np.dot(di, forward_pass["z_s"][t].T)
            b_i_d += di

            # Compute the derivative of the forget gate and update its gradients
            df = dC * C_prev
            df = sigmoid(forward_pass["f_s"][t]) * df
            W_f_d += np.dot(df, forward_pass["z_s"][t].T)
            b_f_d += df

            # Compute the derivative of the input and update the gradients of the previous hidden and cell state
            dz = (np.dot(self.W_f.T, df) + np.dot(self.W_i.T, di) + np.dot(self.W_g.T, dg) + np.dot(self.W_o.T, do))
            dh_prev = dz[:hidden_size, :]
            dC_prev = forward_pass["f_s"][t] * dC

        grads = W_f_d, W_i_d, W_g_d, W_o_d, W_v_d, b_f_d, b_i_d, b_g_d, b_o_d, b_v_d

        # Clip gradients
        grads = self.__clip_gradient_norm(grads)

        return loss, grads


if __name__ == "__main__":
    dataset = Dataset()
    dataset.generate_dataset()
    hidden_size = 1  # Number of dimensions in the hidden state
    lstm = LSTM(hidden_size, dataset.vocab_size)
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

    loss, grads = lstm.backward(forward_pass, targets_one_hot, h, c)

    print('We get a loss of:')
    print(loss)
