from pickle import load
from scripts.data import get_training_data
from scripts.train import train
from scripts.utils import load_pickle

# FOLDER NAMES
# Put your .txt files into data folder and change this line
DATA_PATH = "data/enwiki-preprocessed"

# DO NOT change this line
SAVE_DIR = "cache"

# Read data and create word2index, word_count and sentences
word2index, word_counts, sentences = get_training_data(
    data_path=DATA_PATH,
    save_dir=SAVE_DIR
)

train(
    sentences,
    word2index,
    word_counts,
    SAVE_DIR
)