# import pandas as pd
#  import matplotlib.pyplot as plt
from activation_functions import tanh_activation, sigmoid, softmax
import numpy as np


class LSTM:
    def __init__(self, hidden_size, vocab_size, optimizer, loss_func):
        """
        Initializes our LSTM network.

        Args:
            `hidden_size`: the dimensions of the hidden state (how much we should look back)
            `vocab_size`: the dimensions of our vocabulary
        """
        self.ltm = 0
        self.stm = 0
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self._optimizer = optimizer
        self._loss_func = loss_func
        self.USE_OPTIMIZER = True

        z_size = hidden_size + vocab_size
        self.parameters = {
            "weights": {
                "W_Forget": self.__init_orthogonal(np.zeros((hidden_size, z_size))),
                "W_Input": self.__init_orthogonal(np.zeros((hidden_size, z_size))),
                "W_Candidate": self.__init_orthogonal(np.zeros((hidden_size, z_size))),
                "W_Output": self.__init_orthogonal(np.zeros((hidden_size, z_size))),
                "W_stm": self.__init_orthogonal(np.zeros((vocab_size, hidden_size))),

            },
            "bias": {
                "b_Forget": np.zeros((hidden_size, 1)),
                "b_Input": np.zeros((hidden_size, 1)),
                "b_Candidate": np.zeros((hidden_size, 1)),
                "b_Output": np.zeros((hidden_size, 1)),
                "b_stm": np.zeros((vocab_size, 1)),
            }
        }

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer


    @property
    def loss_func(self):
        return self.loss_func

    @loss_func.setter
    def loss_func(self, loss_func):
        self.loss_func = loss_func


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

        d = np.diag(r, 0)
        ph = np.sign(d)
        q *= ph

        if rows < cols:
            q = q.T

        new_param = q

        return new_param

    @property
    def get_parameters(self):
        """
        Returns weights and biases as 2d array

        """
        return self.parameters

    def forward(self, inputs, stm_prev, ltm_prev):
        """
        Arguments:
            inputs -- your input data at timestep "t", numpy array of shape (n_x, m).
            stm_prev -- g_prev at timestep "t-1", numpy array of shape (n_a, m)
            ltm_prev -- C_prev at timestep "t-1", numpy array of shape (n_a, m)
        Returns:
            z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s -- lists of size m containing the computations in each forward pass
            outputs -- prediction at timestep "t", numpy array of shape (n_v, m)
        """

        # Save a list of computations for each of the components in the LSTM
        forward_pass = {
            "Concat_Input": [],
            "Forget": [],
            "Input": [],
            "Candidate": [],
            "ltm_prev": [],
            "Output": [],
            "stm_prev": [],
            "stm": [],
            "result": []
        }

        # Append the initial cell and hidden state to their respective lists
        forward_pass["stm_prev"].append(stm_prev)
        forward_pass["ltm_prev"].append(ltm_prev)
        for x in inputs:
            # Concatenate input and hidden state
            concat_input = np.row_stack((stm_prev, x))
            forward_pass["Concat_Input"].append(concat_input)
            # Calculate forget gate
            # self.parameters["weights"]["W_Forget"] 300,2800
            # concat_input  301, 1
            print(stm_prev.shape)
            forget_gate = sigmoid(np.dot(self.parameters["weights"]["W_Forget"], concat_input) + self.parameters["bias"]["b_Forget"])
            forward_pass["Forget"].append(forget_gate)

            # Calculate input gate
            input_gate = sigmoid(np.dot(self.parameters["weights"]["W_Input"], concat_input) + self.parameters["bias"]["b_Input"])
            forward_pass["Input"].append(input_gate)

            # Calculate candidate
            candidate = tanh_activation(
                np.dot(self.parameters["weights"]["W_Candidate"], concat_input) + self.parameters["bias"]["b_Candidate"])
            forward_pass["Candidate"].append(candidate)

            # Calculate memory state
            ltm_prev = np.multiply(ltm_prev, forget_gate) + np.multiply(candidate, input_gate)
            forward_pass["ltm_prev"].append(ltm_prev)

            # Calculate output gate
            o = sigmoid(np.dot(self.parameters["weights"]["W_Output"], concat_input) + self.parameters["bias"]["b_Output"])
            forward_pass["Output"].append(o)

            # Calculate hidden state
            stm = o * tanh_activation(ltm_prev)
            forward_pass["stm_prev"].append(stm)

            # Calculate logits
            v = np.dot(self.parameters["weights"]["W_stm"], stm) + self.parameters["bias"]["b_stm"]
            forward_pass["stm"].append(v)

            # Calculate softmax
            result = softmax(v)
            forward_pass["result"].append(result)

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


    def calculate_loss(self, prediction, targets):
        loss = 0
        for t in reversed(range(len(prediction))):
            # Compute the cross entropy
            loss += self.loss_func(prediction[t], targets[t])
        return loss 

    def backward(self, forward_pass, targets):
        """
        Arguments:
        outputs -- your outputs as a list of size m.
        targets -- your targets as a list of size m.
        Returns:
        loss -- crossentropy loss for all elements in output
        grads -- lists of gradients of every element in p
        """

        # Initialize gradients as zero
        gradients = {
            "weights": {
                "W_Forget": np.zeros_like(self.parameters["weights"]["W_Forget"]),
                "W_Input": np.zeros_like(self.parameters["weights"]["W_Input"]),
                "W_Candidate": np.zeros_like(self.parameters["weights"]["W_Candidate"]),
                "W_Output": np.zeros_like(self.parameters["weights"]["W_Output"]),
                "W_stm": np.zeros_like(self.parameters["weights"]["W_stm"]),
            },
            "bias": {
                "b_Forget": np.zeros_like(self.parameters["bias"]["b_Forget"]),
                "b_Input": np.zeros_like(self.parameters["bias"]["b_Input"]),
                "b_Candidate": np.zeros_like(self.parameters["bias"]["b_Candidate"]),
                "b_Output": np.zeros_like(self.parameters["bias"]["b_Output"]),
                "b_stm": np.zeros_like(self.parameters["bias"]["b_stm"]),
            }
        }

        # Set the next cell and hidden state equal to zero
        stm_next = np.zeros_like(forward_pass["stm_prev"][0])  # h
        ltm_next = np.zeros_like(forward_pass["ltm_prev"][0])  # C

        # Track loss
        loss = 0

        for t in reversed(range(len(forward_pass["result"]))):
            # Compute the cross entropy
            loss += self.loss_func(forward_pass["result"][t], targets[t])
            # Get the previous hidden cell state
            ltm_prev = forward_pass["ltm_prev"][t - 1]

            # Compute the derivative of the relation of the hidden-state to the output gate
            dv = np.copy(forward_pass["result"][t])
            dv[np.argmax(targets[t])] -= 1

            # Update the gradient of the relation of the hidden-state to the output gate
            x = np.dot(dv, forward_pass["stm_prev"][t].T)
            gradients["weights"]["W_stm"] += np.dot(dv, forward_pass["stm_prev"][t].T)
            gradients["bias"]["b_stm"] += dv

            # Compute the derivative of the hidden state and output gate
            dh = np.dot(self.parameters["weights"]["W_stm"].T, dv)
            dh += stm_next
            do = dh * tanh_activation(forward_pass["ltm_prev"][t])
            do = sigmoid(forward_pass["Output"][t], derivative=True) * do

            # Update the gradients with respect to the output gate
            gradients["weights"]["W_Output"] += np.dot(do, forward_pass["Concat_Input"][t].T)
            gradients["bias"]["b_Output"] += do

            # Compute the derivative of the cell state and candidate g
            dC = np.copy(ltm_next)
            dC += dh * forward_pass["Output"][t] * tanh_activation(tanh_activation(forward_pass["ltm_prev"][t]),
                                                                   derivative=True)
            dg = dC * forward_pass["Input"][t]
            dg = tanh_activation(forward_pass["Candidate"][t], derivative=True) * dg

            # Update the gradients with respect to the candidate
            gradients["weights"]["W_Candidate"] += np.dot(dg, forward_pass["Concat_Input"][t].T)
            gradients["bias"]["b_Output"] += dg

            # Compute the derivative of the input gate and update its gradients
            di = dC * forward_pass["Candidate"][t]
            di = sigmoid(forward_pass["Input"][t], True) * di
            gradients["weights"]["W_Input"] += np.dot(di, forward_pass["Concat_Input"][t].T)
            gradients["bias"]["b_Input"] += di

            # Compute the derivative of the forget gate and update its gradients
            df = dC * ltm_prev
            df = sigmoid(forward_pass["Forget"][t]) * df
            gradients["weights"]["W_Forget"] += np.dot(df, forward_pass["Concat_Input"][t].T)
            gradients["bias"]["b_Forget"] += df

            # Compute the derivative of the input and update the gradients of the previous hidden and cell state
            dz = (np.dot(self.parameters["weights"]["W_Forget"].T, df) + np.dot(self.parameters["weights"]["W_Input"].T, di) + np.dot(
                self.parameters["weights"]["W_Candidate"].T, dg) + np.dot(self.parameters["weights"]["W_Output"].T, do))

            dh_prev = dz[:self.hidden_size, :]
            dC_prev = forward_pass["Forget"][t] * dC

        # Clip gradients
        grads = self.__clip_gradient_norm(gradients)

        return loss, grads

    def __clip_gradient_norm(self, grads, max_norm=0.25):
        """
        Clips gradients to have a maximum norm of `max_norm`.
        This is to prevent the exploding gradients problem.
        """
        # Set the maximum of the norm to be of type float
        max_norm = float(max_norm)
        total_norm = 0
        # Calculate the L2 norm squared for each gradient and add them to the total norm
        for gate, grad in grads["weights"].items():
            grad_norm = np.sum(np.power(grad, 2))
            total_norm += grad_norm
        total_norm = np.sqrt(total_norm)
        # Calculate clipping coeficient
        clip_coef = max_norm / (total_norm + 1e-6)
        # If the total norm is larger than the maximum allowable norm, then clip the gradient
        if clip_coef < 1:
            for gate, grad in grads["weights"].items():
                grad *= clip_coef
        return grads

    def update_parameters(self, grads, t, lr=0.01):
        # Take a step
        parameters = self.get_parameters["weights"]
        if (self.USE_OPTIMIZER):
            updated_parameters = self.optimizer(parameters=parameters, gradients=grads, learning_rate=lr, t=t)
        else:
            for (_, parameter), (_, grad) in zip(parameters.items(), grads.items()):
                parameter -= lr * grad


