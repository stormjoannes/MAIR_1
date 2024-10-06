""" This file contains the implementation of a Decision Tree to classify dialogue acts."""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from assignment_1a.data_processing import load_data, remove_duplicates, preprocess_data


class DecisionTreeDialogClassifier:
    def __init__(self, filepath):
        self.filepath = filepath
        self.labeled_data = load_data(filepath)
        self.deduplicated_data = remove_duplicates(self.labeled_data)
        self.vectorizer = None
        self.clf_tree = None

    def train_and_evaluate(self, labeled_lines, description):
        X, labels, self.vectorizer = preprocess_data(labeled_lines, method="count")
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.15, random_state=42)

        self.clf_tree = DecisionTreeClassifier(random_state=42, max_depth=20, min_samples_split=5, criterion='entropy')
        self.clf_tree.fit(X_train, y_train)

        y_pred = self.clf_tree.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{description} Accuracy: {accuracy * 100:.2f}%")
        print(classification_report(y_test, y_pred, zero_division=1))

        return accuracy

    def run(self):
        # Calculate accuracy for original data
        print("Evaluating on original data:")
        self.train_and_evaluate(self.labeled_data, "Original Data")

        # Calculate accuracy for deduplicated data
        print("\nEvaluating on deduplicated data:")
        self.train_and_evaluate(self.deduplicated_data, "Deduplicated Data")


# Usage Example
decision_tree_classifier = DecisionTreeDialogClassifier('data/dialog_acts.dat')
decision_tree_classifier.run()

# Expose vectorizer and clf_tree for dialog system
vectorizer = decision_tree_classifier.vectorizer
clf_tree = decision_tree_classifier.clf_tree
