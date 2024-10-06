from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def load_data(filepath):
    """Load data from a file."""
    labeled_lines = []
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        if len(line.strip().split(" ", 1)) == 2:
            original_label, sentence = line.split(" ", 1)
            labeled_lines.append((original_label.strip(), sentence.strip()))
    return labeled_lines


def remove_duplicates(labeled_lines):
    """Remove duplicate sentences from the data."""
    seen_sentences = set()
    return [(label, sentence) for label, sentence in labeled_lines if sentence not in seen_sentences and not seen_sentences.add(sentence)]


def preprocess_data(labeled_lines, method="tfidf"):
    """Convert sentences to features using either TF-IDF or CountVectorizer."""
    sentences = [sentence for _, sentence in labeled_lines]
    if method == "tfidf":
        vectorizer = TfidfVectorizer()
    else:
        vectorizer = CountVectorizer(lowercase=True)
    X = vectorizer.fit_transform(sentences)
    labels = [label for label, _ in labeled_lines]
    return X, labels, vectorizer
