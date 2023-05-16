import numpy as np
from collections import defaultdict


class Sequence:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        # Return the size of the dataset
        return len(self.targets)

    def __getitem__(self, index):
        # Retrieve inputs and targets at the given index
        X = self.inputs[index]
        y = self.targets[index]

        return X, y


def generate_sequences(num_sequences=256):
    """
    Generates a number of sequences as our dataset.

    Args:
        `num_sequences`: the number of sequences to be generated.
    Returns a list of sequences (sentences).

    f.e. sample = "a,a,a,b,b,b,b,EOS"

    """
    samples = []

    for _ in range(num_sequences):
        num_tokens = np.random.randint(1, 9)
        sample = ['a'] * num_tokens + ['b'] * num_tokens + ['EOS']
        samples.append(sample)

    return samples


class Dataset:

    def __init__(self):
        np.random.seed(1337)
        self.word_to_idx = dict()
        self.idx_to_word = dict()
        self.num_sequences = 0
        self.vocab_size = 0
        self.training_set = []
        self.test_set = []
        self.validation_set = []

    def generate_dataset(self, sequences):
        self.__sequences_to_dicts(sequences)
        self.training_set, self.validation_set, self.test_set = self.__create_datasets(sequences)

    def __sequences_to_dicts(self, sequences):
        """
        Creates word_to_idx and idx_to_word dictionaries for a list of sequences.
        f.e.
        {
            a: 1
            b: 2
            EOS: 3
        }
        """

        # Flatten the dataset
        all_words = self.__flatten(sequences)

        # Count number of word occurences in dictionary 
        # TODO can be improved
        word_count = defaultdict(int)
        for word in all_words:
            word_count[word] += 1

        # Sort by frequency
        word_count = sorted(list(word_count.items()), key=lambda sequence: -sequence[1])

        # Create a list of all unique words
        unique_words = [item[0] for item in word_count]

        # Add UNK token to list of words
        unique_words.append('UNK')

        # Count number of sequences and number of unique words
        # TODO can be improved, num_sentences can be saved earlier
        self.num_sentences, self.vocab_size = len(sequences), len(unique_words)

        # Create dictionaries so that we can go from word to index and back
        # If a word is not in our vocabulary, we assign it to token 'UNK'
        self.word_to_idx = defaultdict(lambda: self.vocab_size - 1)
        self.idx_to_word = defaultdict(lambda: 'UNK')

        # Fill dictionaries
        for idx, word in enumerate(unique_words):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

    def __flatten(self, sequences):
        """
        From 2d array makes 1d array

        f.e. [[a,a,b,EOS],[a,a,a,b,b,EOS]] -> [a,a,b,EOS,a,a,a,b,b,EOS]
        """
        return [item for sublist in sequences for item in sublist]

    def __create_datasets(self, sequences, p_train=0.8, p_val=0.1, p_test=0.1):
        # Define partition sizes
        num_train = int(len(sequences) * p_train)
        num_test = int(len(sequences) * p_test)
        num_val = int(len(sequences) * p_val)

        # Split sequences into partitions
        sequences_train = sequences[:num_train]
        sequences_val = sequences[num_train:num_train + num_val]
        sequences_test = sequences[-num_test:]

        def __get_inputs_targets_from_sequences(sequences):
            inputs, targets = [], []

            # Append inputs and targets s.t. both lists contain L-1 words of a sentence of length L
            # but targets are shifted right by one so that we can predict the next word
            for sequence in sequences:
                inputs.append(sequence[:-1])
                targets.append(sequence[1:])

            return inputs, targets

        # Get inputs and targets for each partition
        inputs_train, targets_train = __get_inputs_targets_from_sequences(sequences_train)
        inputs_val, targets_val = __get_inputs_targets_from_sequences(sequences_val)
        inputs_test, targets_test = __get_inputs_targets_from_sequences(sequences_test)

        # Create datasets
        training_set = Sequence(inputs_train, targets_train)
        validation_set = Sequence(inputs_val, targets_val)
        test_set = Sequence(inputs_test, targets_test)

        return training_set, validation_set, test_set

    def one_hot_encode_sequence(self, sequence):
        """
        One-hot encodes a sequence of words given a fixed vocabulary size.

        Args:
            `sentence`: a list of words to encode
            `vocab_size`: the size of the vocabulary

        Returns a 3-D numpy array of shape (num words, vocab size, 1).
        """
        # Encode each word in the sentence
        encoding = np.array([self.__one_hot_encode(self.word_to_idx[word]) for word in sequence])

        # Reshape encoding s.t. it has shape (num words, vocab size, 1)
        encoding = encoding.reshape(encoding.shape[0], encoding.shape[1], 1)

        return encoding  # [[[0],[0],[0],[1.0]],[[1.0],[0],[0],[0]]]

    def __one_hot_encode(self, idx):
        """
        One-hot encodes a single word given its index and the size of the vocabulary.

        Args:
            `idx`: the index of the given word
            `vocab_size`: the size of the vocabulary

        Returns a 1-D numpy array of length `vocab_size`.
        """
        # Initialize the encoded array
        one_hot = np.zeros(self.vocab_size)

        # Set the appropriate element to one
        one_hot[idx] = 1.0

        return one_hot  # 0,0,0,1.0
