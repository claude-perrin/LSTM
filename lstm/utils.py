import numpy as np


def one_hot_encode_sequence(sequence, word_to_idx, vocab_size):
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

    return one_hot  # 0,0,0,1.0
