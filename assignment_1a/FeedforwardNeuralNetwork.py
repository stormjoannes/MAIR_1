"""This file contains the implementation of a Feedforward Neural Network to classify dialogue acts."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_processing import load_data, remove_duplicates


class FeedforwardNeuralNetworkClassifier:
    def __init__(self, filepath, max_words=10000, max_len=128):
        self.data = load_data(filepath)
        self.deduplicated_data = remove_duplicates(self.data)
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = self.create_tokenizer(self.data)
        self.label_encoder = LabelEncoder()

    def create_tokenizer(self, data):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts([sentence for _, sentence in data])
        return tokenizer

    def tokenize_data(self, data):
        sentences = [sentence for _, sentence in data]
        sequences = self.tokenizer.texts_to_sequences(sentences)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.max_len, padding='post')
        return padded_sequences

    def encode_labels(self, data):
        labels = [label for label, _ in data]
        encoded_labels = self.label_encoder.fit_transform(labels)
        return encoded_labels

    def create_model(self):
        model = models.Sequential([
            layers.Embedding(input_dim=self.max_words, output_dim=128, input_length=self.max_len),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(15, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_and_evaluate(self, data, description):
        X = self.tokenize_data(data)
        y = self.encode_labels(data)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = self.create_model()
        model.fit(X_train, y_train, epochs=5, batch_size=16, validation_split=0.2)

        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"{description} Test Accuracy: {test_acc * 100:.2f}%")

    def evaluate(self):
        print("Evaluating on original data:")
        self.train_and_evaluate(self.data, "Original Data")

        print("\nEvaluating on deduplicated data:")
        self.train_and_evaluate(self.deduplicated_data, "Deduplicated Data")


# Usage Example
fnn_classifier = FeedforwardNeuralNetworkClassifier('data/dialog_acts.dat')
fnn_classifier.evaluate()
