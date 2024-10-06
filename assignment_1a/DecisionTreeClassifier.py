""" This file contains the implementation of a Decision Tree to classify dialogue acts."""

data_path = "data/dialog_acts.dat"
def cargar_dataset(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # space split
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                dialog_act, utterance_content = parts
                data.append((dialog_act, utterance_content))
            else:
                print(f"Incorrect line: {line}")
    return data

data = cargar_dataset(data_path)

# imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

#convert text into Bag of Words representation
def preprocess_data(data):
    # sentence
    utterances = [utterance for _, utterance in data]

    # utterances into a Bag of Words representation
    vectorizer = CountVectorizer(lowercase=True)  # Remove stop_words
    X = vectorizer.fit_transform(utterances)  # Transform sentences into BoW
    return X, vectorizer

# Train the Decision Tree classifier
def train_decision_tree_classifier(X_train, y_train):
    # Adjusting hyperparameters to avoid overfitting and improve accuracy
    clf_tree = DecisionTreeClassifier(
        random_state=42,
        max_depth=20,  # Limits the depth of the tree
        min_samples_split=5,  # Minimum samples required to split an internal node
        criterion='entropy'  # Using entropy as the criterion (information gain)
    )
    clf_tree.fit(X_train, y_train)
    return clf_tree

# Evaluate the classifier's performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print(f"Decision Tree model accuracy: {accuracy:.2f}")
    # print(classification_report(y_test, y_pred))

# Interactive classification
def classify_sentence(model, vectorizer):
    while True:
        input_sentence = input("\nEnter a sentence to classify (type 'exit' to stop): ")
        if input_sentence.lower() == 'exit':
            break
        input_bow = vectorizer.transform([input_sentence])
        prediction = model.predict(input_bow)
        print(f"The predicted dialog act is: {prediction[0]}")

# Extract the labels (dialog acts) from the tuples
labels = [dialog_act for dialog_act, _ in data]

# Step 2: Preprocess the data
X, vectorizer = preprocess_data(data)

# Split the data into training and testing sets (increased test size for better generalization)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.15, random_state=42)

# Step 3: Train the Decision Tree classifier
clf_tree = train_decision_tree_classifier(X_train, y_train)

# Step 4: Evaluate the model's performance
evaluate_model(clf_tree, X_test, y_test)

# Interactive sentence classification
# print("\nInteractive test with the Decision Tree model:")
# classify_sentence(clf_tree, vectorizer)

# Accuracy - Training set
# train_predictions = clf_tree.predict(X_train)
# train_accuracy = accuracy_score(y_train, train_predictions)

# Accuracy - Test set
# test_predictions = clf_tree.predict(X_test)
# test_accuracy = accuracy_score(y_test, test_predictions)
#
# print(f"Training Accuracy: {train_accuracy}")
# print(f"Test Accuracy: {test_accuracy}")
