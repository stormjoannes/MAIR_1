""" This file contains the implementation of the Support Vector machine to classify dialogue acts."""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_processing import load_data, remove_duplicates, preprocess_data


class SupportVectorMachineClassifier:
    def __init__(self, filepath):
        self.data = load_data(filepath)
        self.deduplicated_data = remove_duplicates(self.data)

    def train_and_evaluate(self, data, description, output_file):
        X, labels, vectorizer = preprocess_data(data, method="tfidf")
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.15, random_state=42)

        classifier = SVC(C=1.0, kernel='linear')
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{description} Accuracy: {accuracy * 100:.2f}%")

        report = classification_report(y_test, y_pred, zero_division=1)
        cm = confusion_matrix(y_test, y_pred)

        with open(output_file, 'w', encoding='utf-8') as f:
            for original_label, predicted_label in zip(y_test, y_pred):
                f.write(f"{original_label}\t{predicted_label}\n")

        return accuracy

    def run(self):
        print("Original Data Accuracy:")
        self.train_and_evaluate(self.data, "Original Data", "data/original_data_results.txt")

        print("\nDeduplicated Data Accuracy:")
        self.train_and_evaluate(self.deduplicated_data, "Deduplicated Data", "data/deduplicated_data_results.txt")


# Usage Example
svm_classifier = SupportVectorMachineClassifier('data/dialog_acts.dat')
svm_classifier.run()


