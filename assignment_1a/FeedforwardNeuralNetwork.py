"""This file contains the implementation of a Feedforward Neural Network to classify dialogue acts."""

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix as sklearn_confusion_matrix
from utils import retrieve_data


class FeedforwardNeuralNetworkClassifier:
    def __init__(self, filepath, max_words=10000, max_len=128):
        self.data = retrieve_data(filepath)
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = self.create_tokenizer()
        self.le = LabelEncoder()
        self.data['label'] = self.le.fit_transform(self.data['label'])
        self.padded_sequences = self.tokenize_data(self.data)
        self.model = self.create_model()

    def create_tokenizer(self):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(self.data['sentence'])
        return tokenizer

    def tokenize_data(self, data):
        sequences = self.tokenizer.texts_to_sequences(data['sentence'])
        return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.max_len, padding='post')

    def create_model(self):
        model = models.Sequential([
            layers.Embedding(input_dim=self.max_words, output_dim=128, input_length=self.max_len),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(15, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self, validation_split=0.2, epochs=5, batch_size=16):
        train_sentences, test_sentences, train_labels, test_labels = train_test_split(
            self.padded_sequences, self.data['label'], test_size=0.2, random_state=42
        )
        history = self.model.fit(train_sentences, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return history, test_sentences, test_labels

    def evaluate_model(self, test_sentences, test_labels):
        loss, accuracy = self.model.evaluate(test_sentences, test_labels)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    def plot_confusion_matrix(self, test_sentences, test_labels):
        y_pred = np.argmax(self.model.predict(test_sentences), axis=1)
        cm = sklearn_confusion_matrix(test_labels, y_pred)
        class_names = self.le.inverse_transform(np.unique(test_labels))

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def report_classification(self, test_sentences, test_labels):
        y_pred = np.argmax(self.model.predict(test_sentences), axis=1)
        report = classification_report(test_labels, y_pred, target_names=self.le.classes_)
        print(report)


# Usage
fnn_classifier = FeedforwardNeuralNetworkClassifier('data/dialog_acts.dat')
history, test_sentences, test_labels = fnn_classifier.train_model()
fnn_classifier.evaluate_model(test_sentences, test_labels)
fnn_classifier.plot_confusion_matrix(test_sentences, test_labels)
fnn_classifier.report_classification(test_sentences, test_labels)

