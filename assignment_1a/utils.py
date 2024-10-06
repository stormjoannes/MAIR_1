"""This file contains utility functions for the dialogue act classification task."""


def manual_test_model(classify_function, keywords):
    while True:
        sentence = input("Enter a sentence to classify (type 'exit' to stop): ")

        if sentence.lower() == 'exit':
            print("Exiting the classifier.")
            break

        label = classify_function(sentence, keywords)
        print(f"Classified as: {label}")


def retrieve_data(filepath):
    with open(filepath, 'r') as file:
        data = {'label': [], 'sentence': [], 'prediction': []}

        for line in file:
            words = line.split(maxsplit=1)
            if len(words) > 1:
                data['label'].append(words[0])
                data['sentence'].append(words[1])
            else:
                data['label'].append(words[0])
                data['sentence'].append('')
    return data
