"""This file contains the implementation of a Feedforward Neural Network to classify dialogue acts."""

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from utils import manual_test_model, retrieve_data


def tokenize_data(data):
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(data['sentence'])
    sequences = tokenizer.texts_to_sequences(data['sentence'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences


def create_model():
    model = models.Sequential()

    model.add(layers.Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(15, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def model_report(model, sentences, labels):
    predictions = model.predict(sentences)
    predicted_labels = np.argmax(predictions, axis=1)

    all_classes = list(range(15))
    report = classification_report(labels, predicted_labels, labels=all_classes,
                                   target_names=[f'{le.inverse_transform([i])[0]}' for i in all_classes])
    print(report)


def confusion_matrix(sentences, labels):
    # Get the predictions for the test set
    y_pred = model.predict(sentences)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = labels

    cm = sklearn_confusion_matrix(y_true_labels, y_pred_labels)

    class_names = le.inverse_transform(np.unique(y_true_labels))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def classify_sentence_fnn(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=128, padding='post')

    prediction = model.predict(padded_sequence)

    predicted_label_index = np.argmax(prediction, axis=1)[0]
    predicted_label = le.inverse_transform([predicted_label_index])[0]

    return predicted_label


# Load the data
data = retrieve_data("data/dialog_acts.dat")

# Create the first fnn model
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

max_words = 10000  # Maximum number of words
max_len = 128  # Maximum sentence length

padded_sequences = tokenize_data(data)
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    padded_sequences, data['label'], test_size=0.2, random_state=42
)

model = create_model()
model.summary()
history = model.fit(train_sentences, train_labels, epochs=5, batch_size=16, validation_split=0.2)

model_report(model, test_sentences, test_labels)
confusion_matrix(test_sentences, test_labels)

test_loss, test_acc = model.evaluate(test_sentences, test_labels)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

unique_labels = sorted(list(set(data['label'])))
manual_test_model(classify_sentence_fnn)

# ------------------------------- Deduplicate --------------------------------
df = pd.DataFrame(data)
df_cleaned = df.drop_duplicates(subset=['label', 'sentence'])
data_deduplicate = df_cleaned.to_dict(orient='list')
data_deduplicate['label'] = np.array(data_deduplicate['label'])

le_dedup = LabelEncoder()
data_deduplicate['label'] = le_dedup.fit_transform(data_deduplicate['label'])

padded_sequences_dedup = tokenize_data(data_deduplicate)
train_sentences_dedup, test_sentences_dedup, train_labels_dedup, test_labels_dedup = train_test_split(
    padded_sequences_dedup, data_deduplicate['label'], test_size=0.2, random_state=42
)

model_deduplicate = create_model()
history_deduplicate = model_deduplicate.fit(train_sentences_dedup, train_labels_dedup, epochs=5, batch_size=16,
                                            validation_split=0.2)

test_loss_deduplicate, test_acc_deduplicate = model_deduplicate.evaluate(test_sentences_dedup, test_labels_dedup)
print(f"Test Accuracy: {test_acc_deduplicate * 100:.2f}%")

model_report(model, test_sentences_dedup, test_labels_dedup)
confusion_matrix(test_sentences_dedup, test_labels_dedup)
