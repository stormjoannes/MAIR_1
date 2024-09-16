""" This file contains the implementation of the Support Vector machine to classify dialogue acts."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
matplotlib.use('TkAgg')


# Process files: read and classify line by line
def process_file(filepath):
    labeled_lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        for line in lines:
            # Ensure that line contains both label and sentence
            if len(line.strip().split(" ", 1)) == 2:
                # Extract original label and sentence
                original_label, sentence = line.split(" ", 1)
                original_label = original_label.strip()
                sentence = sentence.strip()

                # Store original label and sentence
                labeled_lines.append((original_label, sentence))
            else:
                print(f"Line skipped due to incorrect format: {line.strip()}")

    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"Error while processing file: {e}")

    return labeled_lines


# Function to remove duplicates
def remove_duplicates(labeled_lines):
    seen_sentences = set()
    deduplicated_lines = []

    for label, sentence in labeled_lines:
        if sentence not in seen_sentences:
            deduplicated_lines.append((label, sentence))
            seen_sentences.add(sentence)

    return deduplicated_lines


def save_deduplicated_data(input_file, output_file):
    labeled_lines = process_file(input_file)
    deduplicated_lines = remove_duplicates(labeled_lines)


    with open(output_file, 'w', encoding='utf-8') as file:
        for label, sentence in deduplicated_lines:
            file.write(f"{label} {sentence}\n")

    print(f"Deduplicated data saved to {output_file}")


save_deduplicated_data('data/dialog_acts.dat', 'data/deduplicated_data.txt')


def plot_class_distribution_comparison(original_data, deduplicated_data):
    # Convert data to DataFrames
    df_original = pd.DataFrame(original_data, columns=['label', 'sentence'])
    df_deduplicated = pd.DataFrame(deduplicated_data, columns=['label', 'sentence'])

    # Calculate class distribution for each type
    original_distribution = df_original['label'].value_counts().sort_index()
    deduplicated_distribution = df_deduplicated['label'].value_counts().sort_index()

    # Combine into a single DataFrame
    comparison_df = pd.DataFrame({
        'Original Data': original_distribution,
        'Deduplicated Data': deduplicated_distribution
    }).fillna(0)

    # Plotting
    ax = comparison_df.plot(kind='bar', figsize=(14, 8), width=0.8)
    plt.title('Class Distribution Comparison')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


original_data = process_file('data/dialog_acts.dat')
deduplicated_data = process_file('data/deduplicated_data.txt')  # Ensure this file exists and contains deduplicated data

plot_class_distribution_comparison(original_data, deduplicated_data)


# Function to train and evaluate the model, and save results to a file
def train_and_evaluate(labeled_lines, description, output_file):
    # Extract sentences and labels
    sentences = [sentence for _, sentence in labeled_lines]
    labels = [label for label, _ in labeled_lines]

    # Convert text data into TF-IDF features
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.15, random_state=42)

    # Train SVM classifier
    svm_classifier = SVC(C=1.0, kernel='linear')  # Using linear kernel for simplicity
    svm_classifier.fit(X_train, y_train)

    # Predict labels for testing data
    y_pred = svm_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{description} Accuracy: {accuracy * 100:.2f}%")

    # Calculate confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred, labels=svm_classifier.classes_)
    labels = svm_classifier.classes_  # Use the labels from the classifier
    report = classification_report(y_test, y_pred, labels=labels, target_names=labels, output_dict=True, zero_division=0)

    # Save the test results to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Original Label\tPredicted Label\tSentence\n")
        for original_label, predicted_label, sentence in zip(y_test, y_pred, [sentences[i] for i in X_test.indices]):
            f.write(f"{original_label}\t{predicted_label}\t{sentence}\n")

    print(f"{description} results saved to {output_file}")

    return cm, report, labels


# Get results for original data
cm_orig, report_orig, classes_orig = train_and_evaluate(original_data, "Original Data","data/original_data_results.txt")

# Get results for deduplicated data
cm_dedup, report_dedup, classes_dedup = train_and_evaluate(deduplicated_data, "Deduplicated Data", "data/deduplicated_data_results.txt")

# Ensure that classes are the same for both
assert (classes_orig == classes_dedup).all(), "Class labels should be the same for both datasets."


def plot_confusion_matrices_and_reports(cm_orig, cm_dedup, report_orig, report_dedup, classes):
    # Prepare classification reports
    df_report_orig = pd.DataFrame(report_orig).T
    df_report_dedup = pd.DataFrame(report_dedup).T

    # Remove 'support' column from reports for a cleaner view
    df_report_orig = df_report_orig.drop(columns='support')
    df_report_dedup = df_report_dedup.drop(columns='support')

    # Plot confusion matrices
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
    plt.show()


# Plot the results
plot_confusion_matrices_and_reports(cm_orig, cm_dedup, report_orig, report_dedup, classes_orig)

# Process files and create two datasets
filepath = "data/dialog_acts.dat"  # File path containing original labels and sentences
labeled_output = process_file(filepath)

if labeled_output:
    # Train and evaluate with original data and save results
    train_and_evaluate(labeled_output, "Original Data", "data/original_data_results.txt")

    # Remove duplicates
    deduplicated_output = remove_duplicates(labeled_output)

    # Train and evaluate with deduplicated data and save results
    train_and_evaluate(deduplicated_output, "Deduplicated Data", "data/deduplicated_data_results.txt")
else:
    print("No data to process.")

