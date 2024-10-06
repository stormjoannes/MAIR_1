""" This file contains the implementation of the Support Vector machine to classify dialogue acts."""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_processing import load_data, remove_duplicates, preprocess_data


class SupportVectorMachineClassifier:
    def __init__(self, filepath):
        self.data = load_data(filepath)
        self.deduplicated_data = remove_duplicates(self.data)
        self.vectorizer = None
        self.classifier = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self, data):
        """Train the SVM model."""
        X, labels, self.vectorizer = preprocess_data(data, method="tfidf")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, labels, test_size=0.15, random_state=42)

        self.classifier = SVC(C=1.0, kernel='linear')
        self.classifier.fit(self.X_train, self.y_train)

    def evaluate(self, description, output_file):
        """Evaluate the SVM model."""
        y_pred = self.classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"{description} Accuracy: {accuracy * 100:.2f}%")

        report = classification_report(self.y_test, y_pred, zero_division=1)
        cm = confusion_matrix(self.y_test, y_pred)

        with open(output_file, 'w', encoding='utf-8') as f:
            for original_label, predicted_label in zip(self.y_test, y_pred):
                f.write(f"{original_label}\t{predicted_label}\n")

        return accuracy

    def run(self):
        """Run training and evaluation for both original and deduplicated data."""
        # Train and evaluate on original data
        print("Evaluating on original data:")
        self.train(self.data)
        self.evaluate("Original Data", "data/original_data_results.txt")

        # Train and evaluate on deduplicated data
        print("\nEvaluating on deduplicated data:")
        self.train(self.deduplicated_data)
        self.evaluate("Deduplicated Data", "data/deduplicated_data_results.txt")


# Usage Example
svm_classifier = SupportVectorMachineClassifier('data/dialog_acts.dat')
svm_classifier.run()
