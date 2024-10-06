""" This file contains the implementation of a Decision Tree to classify dialogue acts."""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_processing import load_data, remove_duplicates, preprocess_data


class DecisionTreeDialogClassifier:
    def __init__(self, filepath):
        self.filepath = filepath
        self.labeled_data = load_data(filepath)
        self.deduplicated_data = remove_duplicates(self.labeled_data)
        self.vectorizer = None
        self.clf_tree = None
        self.print_output = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self, labeled_lines):
        """Train the Decision Tree model."""
        X, labels, self.vectorizer = preprocess_data(labeled_lines, method="count")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, labels, test_size=0.15, random_state=42)

        self.clf_tree = DecisionTreeClassifier(random_state=42, max_depth=20, min_samples_split=5, criterion='entropy')
        self.clf_tree.fit(self.X_train, self.y_train)

    def evaluate(self, description):
        """Evaluate the Decision Tree model."""
        y_pred = self.clf_tree.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        if self.print_output:
            print(f"{description} Accuracy: {accuracy * 100:.2f}%")
            print(classification_report(self.y_test, y_pred, zero_division=1))

        return accuracy

    def run(self):
        """Run training and evaluation for both original and deduplicated data."""
        # Train and evaluate on original data
        self.train(self.labeled_data)
        self.evaluate("Original Data")

        # Train and evaluate on deduplicated data
        self.train(self.deduplicated_data)
        self.evaluate("Deduplicated Data")


# Usage Example
decision_tree_classifier = DecisionTreeDialogClassifier('data/dialog_acts.dat')

# Print outputs if running as a script
if __name__ == "__main__":
    decision_tree_classifier.print_output = True
else:
    decision_tree_classifier.print_output = False

decision_tree_classifier.run()

# Expose vectorizer and clf_tree for dialog system
vectorizer = decision_tree_classifier.vectorizer
clf_tree = decision_tree_classifier.clf_tree
