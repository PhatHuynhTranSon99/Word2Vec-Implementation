import numpy as np
import os
import string
import pickle
from scripts.utils import save_pickle

HEADER_SIGNATURE = "[*-|=\{\}"

def is_header_line(line):
    """
    Check if a line is a header line of wiki dataset
    We will not train on the header line
    """
    return len(line) == 0 or line[0] in HEADER_SIGNATURE

def remove_punctuation(line):
    """
    Remove punctuation from a line of text
    """
    return line.translate(str.maketrans('', '', string.punctuation))

def process_line(line):
    """
    Process a line of text to create clean training data
    This will create a list of words
    """
    return remove_punctuation(line).lower().split()

def get_word2index_mapping(data_path, save_dir, unknown_token="UNK", top=20000, save=True):
    """
    Get the word2index mapping from files in a data folder
    """
    # First we count every single word in the training data
    word_count = {}
    total_words = 0
    
    for text_file in os.listdir(data_path):
        with open(os.path.join(data_path, text_file), encoding="utf-8", errors="ignore") as text:
            for line in text:
                # DO NOT process the header lines
                if is_header_line(line):
                    continue

                # Get the words from line using process_line methods
                words = process_line(line)
                total_words += len(words)

                # We count the number of time a word has appeared in the dataset
                for word in words:
                    # If word not in word_count, set initial value to zeross and add 1 to it
                    word_count[word] = word_count.get(word, 0) + 1

    # Then we get only the top 20000 words that appeared in data
    # Sort in reversed order to get the highest occuring words first
    word_count_sorted = sorted(word_count.items(), key=lambda wordCountTuple: wordCountTuple[1], reverse=True)
    top_word_counts = word_count_sorted[:20000]
    
    # Separate the words and the counts
    words = []
    counts = []
    for word, count in top_word_counts:
        words.append(word)
        counts.append(count)

    # Turn count into array
    counts = np.array(counts)

    # Create word2index mapping
    word2index = { unknown_token: 0 }
    for index, word in enumerate(words):
        word2index[word] = index + 1

    # Create word_count vector
    word_count_array = np.zeros((len(word2index),))
    word_count_array[0] = total_words - np.sum(counts) # Count of unknown words = total words - count of known words
    word_count_array[1:] = counts

    # Save the word if save
    if save:
        save_pickle(word2index, os.path.join(save_dir, "word2index.pickle"))
        save_pickle(word_count_array, os.path.join(save_dir, "word_counts.pickle"))

    return word2index, word_count_array


def get_sentences(data_path, save_dir, word2index, save=True):
    """
    Convert sentences in data into list of word,
    if the word is not in word2index, it will be converted to "UNK" token
    """
    sentences = []

    for text_file in os.listdir(data_path):
        with open(os.path.join(data_path, text_file), encoding="utf-8", errors="ignore") as text:
            for line in text:
                # DO NOT process the header lines
                if is_header_line(line):
                    continue

                # Get the words from line using process_line methods
                words = process_line(line)
                processed_words = [word2index[word] if word in word2index else word2index["UNK"] for word in words]

                # Append to sentences
                sentences.append(processed_words)

    if save:
        save_pickle(sentences, os.path.join(save_dir, "sentences.pickle"))

    return sentences

def get_training_data(data_path, save_dir):
    """
    Get the word2vec, word_count and sentences (containing indices)
    from a data folder
    """

    # First get word2index mapping along with the word counts
    word2index, word_counts = get_word2index_mapping(data_path, save_dir)

    # then retrieve the sentences from data
    sentences = get_sentences(data_path, save_dir, word2index)

    return word2index, word_counts, sentences
