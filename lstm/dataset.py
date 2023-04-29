import numpy as np
from collections import defaultdict

# Set seed such that we always get the same dataset
# (this is a good idea in general)
np.random.seed(42)

def generate_dataset(num_sequences=2**8):
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


def flatten(sequences):
    return [item for sublist in sequences for item in sublist]


def sequences_to_dicts(sequences):
    """
    Creates word_to_idx and idx_to_word dictionaries for a list of sequences.
    """
    # A bit of Python-magic to flatten a nested list

    # Flatten the dataset
    all_words = flatten(sequences)

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



class Dataset():
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


def create_datasets(sequences, p_train=0.8, p_test=0.2):
    # Define partition sizes
    num_train = int(len(sequences) * p_train)
    num_test = int(len(sequences) * p_test)

    # Split sequences into partitions
    sequences_train = sequences[:num_train]
    sequences_test = sequences[-num_test:]

    def get_inputs_targets_from_sequences(sequences):
        # Define empty lists
        inputs, targets = [], []

        # Append inputs and targets s.t. both lists contain L-1 words of a sentence of length L
        # but targets are shifted right by one so that we can predict the next word
        for sequence in sequences:
            inputs.append(sequence[:-1])
            targets.append(sequence[1:])

        return inputs, targets

    # Get inputs and targets for each partition
    inputs_train, targets_train = get_inputs_targets_from_sequences(sequences_train)
    inputs_test, targets_test = get_inputs_targets_from_sequences(sequences_test)

    # Create datasets
    training_set = Dataset(inputs_train, targets_train)
    test_set = Dataset(inputs_test, targets_test)

    return training_set, test_set


def one_hot_encode(idx, vocab_size):
    """
    One-hot encodes a single word given its index and the size of the vocabulary.

    Args:
        `idx`: the index of the given word
        `vocab_size`: the size of the vocabulary

    Returns a 1-D numpy array of length `vocab_size`.
    """
    # Initialize the encoded array
    one_hot = np.zeros(vocab_size)

    # Set the appropriate element to one
    one_hot[idx] = 1.0

    return one_hot   # 0,0,0,1.0


def one_hot_encode_sequence(sequence, vocab_size):
    """
    One-hot encodes a sequence of words given a fixed vocabulary size.

    Args:
        `sentence`: a list of words to encode
        `vocab_size`: the size of the vocabulary

    Returns a 3-D numpy array of shape (num words, vocab size, 1).
    """
    # Encode each word in the sentence
    encoding = np.array([one_hot_encode(word_to_idx[word], vocab_size) for word in sequence])

    # Reshape encoding s.t. it has shape (num words, vocab size, 1)
    encoding = encoding.reshape(encoding.shape[0], encoding.shape[1], 1)

    return encoding  # [[[0],[0],[0],[1.0]],[[1.0],[0],[0],[0]]]



if __name__ == "__main__":
    sequences = generate_dataset()  # ['a','a','b','EOS']
    print(f'The sequence[0] example: {sequences[0]}')
    print(f'The sequence[-1] example: {sequences[-1]}')
    word_to_idx, idx_to_word, num_sequences, vocab_size = sequences_to_dicts(sequences)

    print(f'We have {num_sequences} sentences and {len(word_to_idx)} unique tokens in our dataset (including UNK).\n')
    for key, value in word_to_idx.items():
        print(f'The index of {key} is', value)
    print(f'The word corresponding to index 1 is \'{idx_to_word[1]}\'')

    training_set, test_set = create_datasets(sequences)
    print(f'We have {len(training_set)} samples in the training set.')
    print(f'We have {len(test_set)} samples in the test set.')
    print(f'training set inputs[0]: {training_set.inputs[0]}')
    print(f'training set target[0]: {training_set.targets[0]}')
    print(f'training set inputs[1]: {training_set.inputs[1]}')
    print(f'training set target[1]: {training_set.targets[1]}')

    test_word = one_hot_encode(word_to_idx['a'], vocab_size)
    print(f'Our one-hot encoding of \'a\' has shape {test_word.shape}.')


    test_sentence = one_hot_encode_sequence(['a', 'b'], vocab_size)
    print(f'Our one-hot encoding of \'a b\' has shape {test_sentence.shape}.')

