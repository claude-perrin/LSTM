import numpy as np
from collections import defaultdict


class Dataset: 

    def __init__(self):
        np.random.seed(42)
        self.word_to_idx = dict()
        self.idx_to_word = dict()
        self.num_sequences = 0
        self.vocab_size = 0
        self.training_set = []
        self.test_set = []

    class __Sequence():
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

    def __generate_sequences(self,num_sequences=2**8):
        """
        Generates a number of sequences as our dataset.

        Args:
            `num_sequences`: the number of sequences to be generated.

        Returns a list of sequences.
        """
        samples = []

        for _ in range(num_sequences): 
            num_tokens = np.random.randint(1, 12)
            sample = ['a'] * num_tokens + ['b'] * num_tokens + ['EOS']
            samples.append(sample)

        return samples


    def __flatten(self,sequences):
        return [item for sublist in sequences for item in sublist]


    def sequences_to_dicts(self,sequences):
        """
        Creates word_to_idx and idx_to_word dictionaries for a list of sequences.
        """
        # A bit of Python-magic to flatten a nested list

        # Flatten the dataset
        all_words = self.__flatten(sequences)

        # Count number of word occurences
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
        num_sentences, vocab_size = len(sequences), len(unique_words)

        # Create dictionaries so that we can go from word to index and back
        # If a word is not in our vocabulary, we assign it to token 'UNK'
        word_to_idx = defaultdict(lambda: vocab_size - 1)
        idx_to_word = defaultdict(lambda: 'UNK')

        # Fill dictionaries
        for idx, word in enumerate(unique_words):
            # YOUR CODE HERE!
            word_to_idx[word] = idx 
            idx_to_word[idx] = word 

        return word_to_idx, idx_to_word, num_sentences, vocab_size



    def __create_datasets(self, sequences, p_train=0.8, p_test=0.2):
        # Define partition sizes
        num_train = int(len(sequences) * p_train)
        num_test = int(len(sequences) * p_test)

        # Split sequences into partitions
        sequences_train = sequences[:num_train]
        sequences_test = sequences[-num_test:]

        def __get_inputs_targets_from_sequences(sequences):
            # Define empty lists
            inputs, targets = [], []

            # Append inputs and targets s.t. both lists contain L-1 words of a sentence of length L
            # but targets are shifted right by one so that we can predict the next word
            for sequence in sequences:
                inputs.append(sequence[:-1])
                targets.append(sequence[1:])

            return inputs, targets

        # Get inputs and targets for each partition
        inputs_train, targets_train = __get_inputs_targets_from_sequences(sequences_train)
        inputs_test, targets_test = __get_inputs_targets_from_sequences(sequences_test)

        # Create datasets
        training_set = self.__Sequence(inputs_train, targets_train)
        test_set = self.__Sequence(inputs_test, targets_test)

        return training_set, test_set


    def one_hot_encode(self,idx):
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

        return one_hot   # 0,0,0,1.0


    def one_hot_encode_sequence(self, sequence):
        """
        One-hot encodes a sequence of words given a fixed vocabulary size.

        Args:
            `sentence`: a list of words to encode
            `vocab_size`: the size of the vocabulary

        Returns a 3-D numpy array of shape (num words, vocab size, 1).
        """
        # Encode each word in the sentence
        encoding = np.array([self.one_hot_encode(self.word_to_idx[word]) for word in sequence])

        # Reshape encoding s.t. it has shape (num words, vocab size, 1)
        encoding = encoding.reshape(encoding.shape[0], encoding.shape[1], 1)

        return encoding  # [[[0],[0],[0],[1.0]],[[1.0],[0],[0],[0]]]

    def generate_dataset(self):
        sequences = self.__generate_sequences()
        self.word_to_idx, self.idx_to_word, self.num_sequences, self.vocab_size = self.sequences_to_dicts(sequences)
        self.training_set, self.test_set = self.__create_datasets(sequences)

        self.test_word = self.one_hot_encode(self.word_to_idx['a'])
        #  print(f'Our one-hot encoding of \'a\' has shape {test_word.shape}.')
        self.test_sentence = self.one_hot_encode_sequence(['a', 'b'])
        #  print(f'Our one-hot encoding of \'a b\' has shape {test_sentence.shape}.')



if __name__ == "__main__":
    dataset = Dataset()
    dataset.generate_dataset()
    print(dataset.training_set.inputs)
    #  sequences = __generate_sequences()  # ['a','a','b','EOS']
    #  print(f'The sequence[0] example: {sequences[0]}')
    #  print(f'The sequence[-1] example: {sequences[-1]}')
    #
    #  word_to_idx, idx_to_word, num_sequences, vocab_size = self.sequences_to_dicts(sequences)
    #  print(f'We have {num_sequences} sentences and {len(word_to_idx)} unique tokens in our dataset (including UNK).\n')
    #  for key, value in word_to_idx.items():
    #      print(f'The index of {key} is', value)
    #  print(f'The word corresponding to index 1 is \'{idx_to_word[1]}\'')

    #  training_set, test_set = create_datasets(sequences)
    #  print(f'We have {len(training_set)} samples in the training set.')
    #  print(f'We have {len(test_set)} samples in the test set.')
    #  print(f'training set inputs[0]: {training_set.inputs[0]}')
    #  print(f'training set target[0]: {training_set.targets[0]}')
    #  print(f'training set inputs[1]: {training_set.inputs[1]}')
    #  print(f'training set target[1]: {training_set.targets[1]}')
    #
    #  test_word = one_hot_encode(word_to_idx['a'], vocab_size)
    #  print(f'Our one-hot encoding of \'a\' has shape {test_word.shape}.')
    #
    #
    #  test_sentence = one_hot_encode_sequence(['a', 'b'], vocab_size)
    #  print(f'Our one-hot encoding of \'a b\' has shape {test_sentence.shape}.')

