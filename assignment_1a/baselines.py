"""This file contains the baselines for the dialogue act classification task."""

import json
from utils import manual_test_model, retrieve_data


class BaselineClassifier:
    def __init__(self, filepath, keywords_file):
        self.data = retrieve_data(filepath)
        self.keywords = self.load_keywords(keywords_file)

    def load_keywords(self, filepath):
        with open(filepath, 'r') as file:
            return json.load(file)

    def count_labels(self):
        label_counts = {}
        for label in self.data['label']:
            label_counts[label] = label_counts.get(label, 0) + 1
        for label, count in label_counts.items():
            print(f"{label}: {count}")

    def classify_sentence(self, sentence, keywords=None):
        if keywords is None:
            keywords = self.keywords  # Fallback to default if not provided
        sentence_lower = sentence.lower()
        sorted_keywords = sorted(((label, keyword) for label, kw_list in keywords.items() for keyword in kw_list),
                                 key=lambda x: len(x[1]), reverse=True)
        for label, keyword in sorted_keywords:
            if keyword in sentence_lower:
                return label
        return 'null'

    def apply_baseline(self, default_label="inform"):
        self.data['prediction'] = [default_label] * len(self.data['sentence'])
        accuracy = self.calculate_accuracy(self.data['label'], self.data['prediction'])
        print(f"Inform baseline accuracy: {accuracy:.2f}%")
        self.count_labels()

    def apply_keyword_model(self):
        self.data['prediction'] = [self.classify_sentence(sentence) for sentence in self.data['sentence']]
        accuracy = self.calculate_accuracy(self.data['label'], self.data['prediction'])
        print(f"Keyword model accuracy: {accuracy:.2f}%")

    @staticmethod
    def calculate_accuracy(labels, predictions):
        correct = sum(1 for l, p in zip(labels, predictions) if l == p)
        total = len(labels)
        return (correct / total) * 100 if total > 0 else 0

    def manual_test(self):
        manual_test_model(self.classify_sentence, self.keywords)


# Usage Example
baseline_classifier = BaselineClassifier("../data/dialog_acts.dat", "../data/keywords.json")
baseline_classifier.apply_baseline()
baseline_classifier.apply_keyword_model()
baseline_classifier.manual_test()

