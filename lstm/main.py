from lstm import LSTM
from dataset import Dataset, generate_sequences
import numpy as np
import matplotlib.pyplot as plt


def train(dataset, hidden_size, vocab_size):

    # Hyper-parameters
    num_epochs = 100

    # Initialize a new network
    lstm = LSTM(hidden_size=hidden_size, vocab_size=vocab_size)

    # Initialize hidden state as zeros
   #hidden_state = np.zeros((hidden_size, 1))

    # Track loss
    training_loss, validation_loss = [], []

    # For each epoch
    for i in range(num_epochs):

        # Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0

        # For each sentence in validation set
        for inputs, targets in dataset.validation_set:

            # One-hot encode input and target sequence
            inputs_one_hot = dataset.one_hot_encode_sequence(inputs)
            targets_one_hot = dataset.one_hot_encode_sequence(targets)

            # Initialize hidden state and cell state as zeros
            h = np.zeros((hidden_size, 1))
            c = np.zeros((hidden_size, 1))

            # Forward pass
            forward_pass = lstm.forward(inputs_one_hot, h, c)

            # Backward pass
            loss, _ = lstm.backward(forward_pass, targets_one_hot)

            # Update loss
            epoch_validation_loss += loss
        # For each sentence in training set
        t = 0
        for inputs, targets in dataset.training_set:

            # One-hot encode input and target sequence
            inputs_one_hot = dataset.one_hot_encode_sequence(inputs)
            targets_one_hot = dataset.one_hot_encode_sequence(targets)

            # Initialize hidden state and cell state as zeros
            h = np.zeros((hidden_size, 1))
            c = np.zeros((hidden_size, 1))

            # Forward pass
            forward_pass = lstm.forward(inputs_one_hot, h, c)

            # Backward pass
            loss, grads = lstm.backward(forward_pass, targets_one_hot)

            # Update parameters

            params = lstm.update_parameters(grads)

            # Update loss
            #output_sentence = [dataset.idx_to_word[np.argmax(output)] for output in forward_pass["output_s"]]

            epoch_training_loss += loss

        # Save loss for plot
        training_loss.append(epoch_training_loss / len(dataset.training_set))
        validation_loss.append(epoch_validation_loss / len(dataset.validation_set))

        # Print loss every 10 epochs
        if i % 10 == 0:
            print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')
            print(f'Input sentence {i}:')
            print(inputs)

            print(f'\nTarget sequence {i}:')
            print(targets)

            print('\nPredicted sequence:')
            print([dataset.idx_to_word[np.argmax(output)] for output in forward_pass["result"]])
    return training_loss, validation_loss


if __name__ == "__main__":
    #  sequences = []
    #
    #  with open("data.txt", "r") as file:
    #      for line in file:
    #          words = line.split()
    #          sequences.append(words)
    #  print(sequences)
    hidden_size = 24
    sequences = generate_sequences()
    # print(sequences)
    dataset = Dataset()
    dataset.generate_dataset(sequences)

    training_loss, validation_loss = train(dataset,hidden_size=hidden_size,vocab_size=dataset.vocab_size)
    epoch = np.arange(len(training_loss))
    plt.figure()
    plt.plot(epoch, training_loss, 'r', label='Training loss',)
    plt.plot(epoch, validation_loss, 'b', label='Validation loss')
    plt.legend()
    plt.xlabel('Epoch'), plt.ylabel('MeanSquare loss')
    plt.show()
