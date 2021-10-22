import numpy as np
import os
from scripts.utils import save_pickle, sigmoid


def calculate_negative_sampling_distribution(word_counts):
    """
    Calculate the probability to sample word for negative samples
    """
    return (word_counts ** 0.75) / np.sum(word_counts ** 0.75)


def calculate_drop_distribution(neg_sampling_distribution, threshold=1e-5):
    """
    Calculate the dropout distribution for word in a sentence
    """
    return 1 - np.sqrt(threshold / neg_sampling_distribution)


def subsample(sentence, drop_distribution):
    """
    Drop out random element from the sentence using calculated drop_distribution
    """
    return [word for word in sentence if np.random.random() < 1 - drop_distribution[word]]


def get_context_words(sentence, center_word_position, window_size=5):
    """
    Get the context window for a word in a sentence based on the window size
    """
    window_start_position = max(0, center_word_position - window_size)
    window_end_position = min(len(sentence) - 1, center_word_position + window_size)

    context = []
    for position in range(window_start_position, window_end_position + 1):
        if position != center_word_position:
            context.append(sentence[position])

    return context


def negative_sampling_loss_and_gradient(centerWordIndex, positiveWordIndices, negativeWordIndices, U, V, learning_rate):
    """
    Calculate the loss and perform gradient descent on center word, positive examples and negative examples
    """
    
    # Get the vectors
    centerWordVector = V[centerWordIndex]
    positiveWordVectors = U[positiveWordIndices]
    negativeWordVectors = U[negativeWordIndices]

    # Calculate the logits from positive and negatove vectors
    positive_logits = sigmoid(positiveWordVectors @ centerWordVector)   #sigmoid
    negative_logits = sigmoid(-negativeWordVectors @ centerWordVector)  #1 - sigmoid

    # Calculate the loss
    loss = - np.sum(np.log(positive_logits)) - np.sum(np.log(negative_logits))

    # Calculate the gradient
    centerWordGradient = - (1 - positive_logits) @ positiveWordVectors - (negative_logits - 1) @ negativeWordVectors
    postiveWordGradient = np.outer(positive_logits - 1, centerWordVector)
    negativeWordGradient = np.outer(1 - negative_logits, centerWordVector)

    # Gradient descent
    V[centerWordIndex] -= learning_rate * centerWordGradient
    U[positiveWordIndices] -= learning_rate * postiveWordGradient
    U[negativeWordIndices] -= learning_rate * negativeWordGradient

    return loss


def get_negative_samples(negative_sample_distribution, total=5):
    return np.random.choice(len(negative_sample_distribution), size=total, p=negative_sample_distribution)


def train(sentences, word2index, word_counts, save_dir, start_learning_rate=0.025, end_learning_rate=0.0001, epochs=20, negative_samples=5, D=50):
    """
    Main loop to train word2vec models
    """
    # Create U and V matrices
    U = np.random.randn(len(word2index), D)
    V = np.random.randn(len(word2index), D)

    # Calculate the negative sampling distribution
    p_negative_sampling = calculate_negative_sampling_distribution(word_counts)

    # Calculate the drop out distribution
    p_drop = calculate_drop_distribution(p_negative_sampling)

    # List of cost
    losses = []

    # Learning rate decays
    learning_rate_delta = (start_learning_rate - end_learning_rate) / epochs
    learning_rate = start_learning_rate

    # Start training
    for epoch in range(epochs):
        # Hold the current loss
        current_loss = 0

        # Shuffle sentences randomly
        np.random.shuffle(sentences)

        # Train on each sentences
        for index, sentence in enumerate(sentences):
            # Show progress
            if (index + 1) % 100000 == 0:
                print(f"Epoch {epoch + 1}, Sentence {index + 1}")

            # Skip short sentences
            if len(sentence) < 2:
                continue

            # Perform subsamplimg
            sentence = subsample(sentence, p_drop)

            # Loop through each word in the sentence
            for index, centerWordIndex in enumerate(sentence):
                # Get positive samples
                positive_word_indices = get_context_words(sentence, index)

                # Get negative samples
                negative_word_indices = get_negative_samples(p_negative_sampling, total=negative_samples)

                # Perform gradient descent
                loss = negative_sampling_loss_and_gradient(
                    centerWordIndex, 
                    positive_word_indices, 
                    negative_word_indices,
                    U,
                    V,
                    learning_rate=learning_rate)

                current_loss += loss

        # Display the loss
        print(f"Epoch {epoch + 1}, Loss: {current_loss}")

        # Append the loss
        losses.append(current_loss)

        # Update learning rate
        learning_rate -= learning_rate_delta

        # Save U and V
        save_pickle(U, os.path.join(save_dir, "U.pickle"))
        save_pickle(V, os.path.join(save_dir, "V.pickle"))

    return U, V


