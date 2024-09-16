"""This file contains the baselines for the dialogue act classification task."""

import json
from utils import manual_test_model, retrieve_data

data = retrieve_data("data/dialog_acts.dat")

for sentence in data["sentence"]:
    data["prediction"].append("inform")


correct = 0
total = len(data['label'])

for i in range(total):
    if data['label'][i] == data['prediction'][i]:
        correct += 1

accuracy = correct / total * 100
print(f"Inform baseline accuracy: {accuracy:.2f}%\n")


def count_labels(data):
    label_counts = {}

    # Count the occurrences of each label
    for label in data['label']:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    # Print the count of each label
    for label, count in label_counts.items():
        print(f"{label}: {count}")


count_labels(data)

# Open and load the JSON file
with open('data/keywords.json', 'r') as file:
    keywords = json.load(file)


def classify_sentence(sentence, keywords):
    # Convert the sentence to lowercase
    sentence_lower = sentence.lower()

    # Sort keywords by length in descending order to match longer phrases first
    sorted_keywords = sorted(((label, keyword) for label, kw_list in keywords.items() for keyword in kw_list),
                             key=lambda x: len(x[1]), reverse=True)

    # Check if any keyword is in the sentence
    for label, keyword in sorted_keywords:
        if keyword in sentence_lower:
            return label

    # Return 'null' if no keywords are found
    return 'null'


data['prediction'] = [classify_sentence(sentence, keywords) for sentence in data['sentence']]


def calculate_accuracy_filtered(labels, predictions):
    filtered_labels = []
    filtered_predictions = []

    for label, prediction in zip(labels, predictions):
        filtered_labels.append(label)
        filtered_predictions.append(prediction)

    if len(filtered_labels) != len(filtered_predictions):
        raise ValueError("Filtered labels and predictions lists must be of the same length.")

    correct = sum(1 for l, p in zip(filtered_labels, filtered_predictions) if l == p)
    total = len(filtered_labels)
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy


accuracy = calculate_accuracy_filtered(data['label'], data['prediction'])
print("\nKeywords model baseline accuracy: ", accuracy, "\n")

from collections import defaultdict


def calculate_accuracy_per_label(labels, predictions):
    label_counts = defaultdict(int)
    correct_counts = defaultdict(int)

    for label, prediction in zip(labels, predictions):
        label_counts[label] += 1
        if label == prediction:
            correct_counts[label] += 1

    accuracy_per_label = {}
    for label in label_counts:
        total = label_counts[label]
        correct = correct_counts[label]
        accuracy_per_label[label] = (correct / total) * 100 if total > 0 else 0

    return accuracy_per_label


accuracy_per_label = calculate_accuracy_per_label(data['label'], data['prediction'])
for label, accuracy in accuracy_per_label.items():
    print(f"{label}, Accuracy: {accuracy:.2f}%")


def classify_sentence(sentence):
    for label, words in keywords.items():
        if any(word in sentence.lower() for word in words):
            return label
    return 'Unknown'  # Default label if no keywords are matched


manual_test_model(classify_sentence)
