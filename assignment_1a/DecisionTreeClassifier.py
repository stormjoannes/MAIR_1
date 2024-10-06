""" This file contains the implementation of a Decision Tree to classify dialogue acts."""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


class DecisionTreeDialogClassifier:
    def __init__(self, filepath):
        self.filepath = filepath
        self.labeled_data = self.load_data()
        self.vectorizer = None
        self.clf_tree = None

    def load_data(self):
        data = []
        with open(self.filepath, 'r') as file:
            for line in file:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    dialog_act, sentence = parts
                    data.append((dialog_act, sentence))
        return data

    def preprocess_data(self):
        sentences = [sentence for _, sentence in self.labeled_data]
        self.vectorizer = CountVectorizer(lowercase=True)
        X = self.vectorizer.fit_transform(sentences)
        return X

    def train_classifier(self, X, y):
        clf_tree = DecisionTreeClassifier(random_state=42, max_depth=20, min_samples_split=5, criterion='entropy')
        clf_tree.fit(X, y)
        return clf_tree

    def evaluate_classifier(self, clf, X_test, y_test):
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # Set zero_division=1 to prevent the warning and treat undefined metrics as 1
        report = classification_report(y_test, y_pred, zero_division=1)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(report)
        return report

    def run(self):
        X = self.preprocess_data()
        labels = [dialog_act for dialog_act, _ in self.labeled_data]
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.15, random_state=42)
        self.clf_tree = self.train_classifier(X_train, y_train)
        self.evaluate_classifier(self.clf_tree, X_test, y_test)


# Usage Example
decision_tree_classifier = DecisionTreeDialogClassifier("data/dialog_acts.dat")
decision_tree_classifier.run()

# Ensure vectorizer and clf_tree are accessible
vectorizer = decision_tree_classifier.vectorizer
clf_tree = decision_tree_classifier.clf_tree
