from lstm import LSTM
from dataset import Dataset, generate_sequences
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from optimizer import adam_optimizer as Adam
#from utils import one_hot_encode_sequence


def my_build_model(X, hidden_size, input_size):
    lstm = LSTM(hidden_size = hidden_size, input_size = input_size, optimizer=Adam, loss_func=mean_squared_error)
    return lstm


def train_split(X, Y, test_size):
    length = int(len(X) * test_size)
    X_train = X[1:length]
    X_valid = X[length:]
    Y_train = Y[1:length]
    Y_valid = Y[length:]
    return X_train, X_valid, Y_train, Y_valid


def my_train(lstm, X, Y, hidden_size, input_size):

    # Hyper-parameters
    num_epochs = 100

    # Initialize hidden state as zeros
    # hidden_state = np.zeros((hidden_size, 1))

    X_train, X_valid, Y_train, Y_valid = train_split(X = X, Y = Y, test_size = 0.20)

    # Track loss
    training_loss, validation_loss = [], []

    # For each epoch
    for i in range(num_epochs):

        # Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0

        # For each sentence in validation set
        for inputs, targets in zip(X_valid, Y_valid):

            # Initialize hidden state and cell state as zeros
            h = np.zeros((hidden_size, 1))
            c = np.zeros((hidden_size, 1))
            # Forward pass
            forward_pass = lstm.forward(inputs, h, c)

            # Backward pass
            loss = lstm.calculate_loss(forward_pass["result"][-1], targets)

            # Update loss
            epoch_validation_loss += loss

        # For each sentence in training set
        t = 1
        for inputs, targets in zip(X_train, Y_train):

            # One-hot encode input and target sequence
            #  inputs_one_hot = dataset.one_hot_encode_sequence(inputs)
            #  targets_one_hot = dataset.one_hot_encode_sequence(targets)

            # Initialize hidden state and cell state as zeros
            h = np.zeros((hidden_size, 1))
            c = np.zeros((hidden_size, 1))

            # Forward pass
            forward_pass = lstm.forward(inputs, h, c)

            # Backward pass
            loss, grads = lstm.backward(forward_pass, targets)

            # Update parameters

            params = lstm.update_parameters(grads=grads["weights"], t=t)
            t += 1
            # Update loss
            #output_sentence = [dataset.idx_to_word[np.argmax(output)] for output in forward_pass["output_s"]]

            epoch_training_loss += loss

        # Save loss for plot
        training_loss.append(epoch_training_loss / len(X_train))
        validation_loss.append(epoch_validation_loss / len(X_valid))

        # Print loss every 10 epochs
        if i % 10 == 0:
            print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')
            print(f'Input sentence {i}:')
            print(inputs)

            print(f'\nTarget sequence {i}:')
            print(targets)

            print('\nPredicted sequence:')
            print([np.argmax(output)] for output in forward_pass["result"])
    return training_loss, validation_loss


if __name__ == "__main__":
    embed_dim = 12
    lstm_out = 300
    batch_size = 32
    input_size = 2500
    X = [[0,0,9,6,324,2,4,131,289,109,293,9],[2,84,67,60,74,97, 4,667,388,554,67,46,15]]
    Y = [[1], [0]]
    model = my_build_model(X, hidden_size=lstm_out, input_size = len(X[0]))
    my_train(model, X, Y, lstm_out, input_size)

