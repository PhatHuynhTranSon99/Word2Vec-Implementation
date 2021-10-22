from pickle import load
from scripts.data import get_word2index_mapping, get_sentences
from scripts.train import train
from scripts.utils import load_pickle

word2index = load_pickle("cache/word2index.pickle")
sentences = load_pickle("cache/sentences.pickle")
word_counts = load_pickle("cache/word_counts.pickle")

train(
    sentences,
    word2index,
    word_counts,
    "cache"
)