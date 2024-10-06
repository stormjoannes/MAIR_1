""" This file contains the implementation of the Support Vector machine to classify dialogue acts."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class SupportVectorMachineClassifier:
    def __init__(self, filepath):
        self.filepath = filepath
        self.labeled_lines = self.process_file(filepath)

    @staticmethod
    def process_file(filepath):
        labeled_lines = []
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            for line in lines:
                if len(line.strip().split(" ", 1)) == 2:
                    original_label, sentence = line.split(" ", 1)
                    labeled_lines.append((original_label.strip(), sentence.strip()))
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"Error while processing file: {e}")
        return labeled_lines

    @staticmethod
    def remove_duplicates(labeled_lines):
        seen_sentences = set()
        return [(label, sentence) for label, sentence in labeled_lines if sentence not in seen_sentences and not seen_sentences.add(sentence)]

    def save_deduplicated_data(self, output_file):
        deduplicated_lines = self.remove_duplicates(self.labeled_lines)
        with open(output_file, 'w', encoding='utf-8') as file:
            for label, sentence in deduplicated_lines:
                file.write(f"{label} {sentence}\n")
        print(f"Deduplicated data saved to {output_file}")

    def train_and_evaluate(self, labeled_lines, description, output_file):
        sentences = [sentence for _, sentence in labeled_lines]
        labels = [label for label, _ in labeled_lines]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.15, random_state=42)

        classifier = SVC(C=1.0, kernel='linear')
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{description} Accuracy: {accuracy * 100:.2f}%")

        report = classification_report(y_test, y_pred, labels=classifier.classes_, target_names=classifier.classes_, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)

        with open(output_file, 'w', encoding='utf-8') as f:
            for original_label, predicted_label, sentence in zip(y_test, y_pred, [sentences[i] for i in X_test.indices]):
                f.write(f"{original_label}\t{predicted_label}\t{sentence}\n")

        return cm, report, classifier.classes_

    @staticmethod
    def plot_class_distribution_comparison(original_data, deduplicated_data):
        df_original = pd.DataFrame(original_data, columns=['label', 'sentence'])
        df_deduplicated = pd.DataFrame(deduplicated_data, columns=['label', 'sentence'])

        original_distribution = df_original['label'].value_counts().sort_index()
        deduplicated_distribution = df_deduplicated['label'].value_counts().sort_index()

        comparison_df = pd.DataFrame({
            'Original Data': original_distribution,
            'Deduplicated Data': deduplicated_distribution
        }).fillna(0)

        ax = comparison_df.plot(kind='bar', figsize=(14, 8), width=0.8)
        plt.title('Class Distribution Comparison')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()  # Ensure the plot is displayed

    @staticmethod
    def plot_confusion_matrices_and_reports(cm_orig, cm_dedup, report_orig, report_dedup, classes):
        # Prepare classification reports
        df_report_orig = pd.DataFrame(report_orig).T
        df_report_dedup = pd.DataFrame(report_dedup).T

        # Remove 'support' column from reports for a cleaner view
        df_report_orig = df_report_orig.drop(columns='support')
        df_report_dedup = df_report_dedup.drop(columns='support')

        # Plot confusion matrices and reports
        plt.figure(figsize=(16, 12))

        plt.subplot(2, 2, 1)
        sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix - Original Data')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        plt.subplot(2, 2, 2)
        sns.heatmap(cm_dedup, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix - Deduplicated Data')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Plot classification reports
        plt.subplot(2, 2, 3)
        sns.heatmap(df_report_orig, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Classification Report - Original Data')
        plt.xticks(rotation=45)

        plt.subplot(2, 2, 4)
        sns.heatmap(df_report_dedup, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Classification Report - Deduplicated Data')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()  # Ensure the plot is displayed


# Usage Example
svm_classifier = SupportVectorMachineClassifier('data/dialog_acts.dat')
svm_classifier.save_deduplicated_data('data/deduplicated_data.txt')
cm_orig, report_orig, classes_orig = svm_classifier.train_and_evaluate(svm_classifier.labeled_lines, "Original Data", "data/original_data_results.txt")
deduplicated_lines = svm_classifier.remove_duplicates(svm_classifier.labeled_lines)
cm_dedup, report_dedup, _ = svm_classifier.train_and_evaluate(deduplicated_lines, "Deduplicated Data", "data/deduplicated_data_results.txt")

# Plotting the comparison
svm_classifier.plot_class_distribution_comparison(svm_classifier.labeled_lines, deduplicated_lines)
svm_classifier.plot_confusion_matrices_and_reports(cm_orig, cm_dedup, report_orig, report_dedup, classes_orig)
