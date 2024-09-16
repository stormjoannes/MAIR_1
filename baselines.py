"""This file contains the baselines for the dialogue act classification task."""

import json
from utils import manual_test_model, retrieve_data


def count_labels(data):
    """Count the occurrences of each label in the dataset."""
    label_counts = {}

    for label in data['label']:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    for label, count in label_counts.items():
        print(f"{label}: {count}")


def classify_sentence(sentence, keywords):
    """Classify a sentence based on the keywords provided."""
    sentence_lower = sentence.lower()

    sorted_keywords = sorted(((label, keyword) for label, kw_list in keywords.items() for keyword in kw_list),
                             key=lambda x: len(x[1]), reverse=True)

    for label, keyword in sorted_keywords:
        if keyword in sentence_lower:
            return label

    # Classify 'null' if no keywords are found
    return 'null'


def calculate_total_accuracy(labels, predictions):
    """Calculate the accuracy of the model using the labels and predictions."""

    correct = sum(1 for l, p in zip(labels, predictions) if l == p)
    total = len(labels)
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy


def calculate_accuracy_per_label(labels, predictions):
    """Calculate the accuracy per label using the labels and predictions."""
    label_counts = {}
    correct_counts = {}

    for label, prediction in zip(labels, predictions):
        if label not in label_counts:
            label_counts[label] = 0
        if label not in correct_counts:
            correct_counts[label] = 0

        label_counts[label] += 1
        if label == prediction:
            correct_counts[label] += 1

    accuracy_per_label = {}
    for label in label_counts:
        total = label_counts[label]
        correct = correct_counts[label]
        accuracy_per_label[label] = (correct / total) * 100 if total > 0 else 0

    return accuracy_per_label


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

count_labels(data)

with open('data/keywords.json', 'r') as file:
    keywords = json.load(file)

data['prediction'] = [classify_sentence(sentence, keywords) for sentence in data['sentence']]
accuracy = calculate_total_accuracy(data['label'], data['prediction'])
print("\nKeywords model baseline accuracy: ", accuracy, "\n")

accuracy_per_label = calculate_accuracy_per_label(data['label'], data['prediction'])
for label, accuracy in accuracy_per_label.items():
    print(f"{label}, Accuracy: {accuracy:.2f}%")

manual_test_model(classify_sentence)
