from collections import Counter

# sample sentences taken from blackboard
blackboard_sentences = [
    "I'm looking for world food",
    "I want a restaurant that serves world food",
    "I want a restaurant serving Swedish food",
    "I'm looking for a restaurant in the center",
    "I would like a cheap restaurant in the west part of town",
    "I'm looking for a moderately priced restaurant in the west part of town",
    "I'm looking for a restaurant in any area that serves Tuscan food",
    "Can I have an expensive restaurant",
    "I'm looking for an expensive restaurant and it should serve international food",
    "I need a Cuban restaurant that is moderately priced",
    "I'm looking for a moderately priced restaurant with Catalan food",
    "What is a cheap restaurant in the south part of town",
    "What about Chinese food",
    "I wanna find a cheap restaurant",
    "I'm looking for Persian food please",
    "Find a Cuban restaurant in the center"
]

# Recursive Levenshtein distance function
def levenshtein_recursive(str1, str2, m, n):
    if m == 0:
        return n
    if n == 0:
        return m
    if str1[m - 1] == str2[n - 1]:
        return levenshtein_recursive(str1, str2, m - 1, n - 1)
    return 1 + min(
        levenshtein_recursive(str1, str2, m, n - 1),  # Insert
        levenshtein_recursive(str1, str2, m - 1, n),  # Remove
        levenshtein_recursive(str1, str2, m - 1, n - 1)  # Replace
    )

# Basic dictionary for categorization
basic_dict = {
    'food_type': ['asian', 'korean', 'japanese', 'indian', 'thai', 'cuban', 'french', 'mexican', 'western'],
    'price_range': ['moderately', 'expensive', 'cheap'],
    'location': ['north', 'south', 'east', 'west', 'center', 'any']
}

# Dynamic dictionary to update with new findings
dynamic_dict = {
    'food_type': set(),
    'price_range': set(),
    'location': set()
}

# Stopwords
stopwords = set(
    ['a', 'an', 'the', 'in', 'that', 'priced', 'would']
)

word_counter = Counter()

# Function to find closest match using Levenshtein distance
def apply_levenshtein(word, category_words, threshold=1):
    closest_word = None
    min_distance = float('inf')
    for new_word in category_words:
        distance = levenshtein_recursive(word, new_word, len(word), len(new_word))
        if distance < min_distance and distance <= threshold:
            closest_word = new_word
            min_distance = distance
    return closest_word

# Function to categorize words from the sentences
def categorize_words(sentence, basic_dict, dynamic_dict):
    sentence_words = sentence.lower().split()
    categories = []
    categorized_words = set()

    # Check for "in the" location pattern
    for i in range(1, len(sentence_words) - 1):
        if sentence_words[i - 1:i + 1] == ['in', 'the']:
            location_word = sentence_words[i + 1]
            if location_word in basic_dict['location']:
                categories.append((location_word, 'location'))
                categorized_words.add(location_word)
            elif location_word in dynamic_dict['location']:
                categories.append((location_word, 'location'))
                categorized_words.add(location_word)

    # Check for "food" pattern
    for i in range(1, len(sentence_words)):
        if sentence_words[i] == 'food' and sentence_words[i - 1] not in stopwords:
            food_type_word = sentence_words[i - 1]
            if food_type_word in basic_dict['food_type']:
                categories.append((food_type_word, 'food_type'))
                categorized_words.add(food_type_word)
            elif food_type_word in dynamic_dict['food_type']:
                categories.append((food_type_word, 'food_type'))
                categorized_words.add(food_type_word)
            else:
                dynamic_dict['food_type'].add(food_type_word)
                categories.append((food_type_word, 'food_type'))
                categorized_words.add(food_type_word)

    # Check for "priced" pattern
    for i in range(2, len(sentence_words)):
        if sentence_words[i] == 'priced' and sentence_words[i - 1] not in stopwords:
            price_word = sentence_words[i - 1]
            if price_word in basic_dict['price_range']:
                categories.append((price_word, 'price_range'))
                categorized_words.add(price_word)
            elif price_word in dynamic_dict['price_range']:
                categories.append((price_word, 'price_range'))
                categorized_words.add(price_word)
            else:
                dynamic_dict['price_range'].add(price_word)
                categories.append((price_word, 'price_range'))
                categorized_words.add(price_word)

    # Check for "restaurant" pattern
    for i in range(1, len(sentence_words)):
        if sentence_words[i] == 'restaurant' and sentence_words[i - 1] not in stopwords:
            new_word = sentence_words[i - 1]
            if new_word in basic_dict['food_type'] or new_word in dynamic_dict['food_type']:
                categories.append((new_word, 'food_type'))
                categorized_words.add(new_word)
            elif new_word in basic_dict['price_range'] or new_word in dynamic_dict['price_range']:
                categories.append((new_word, 'price_range'))
                categorized_words.add(new_word)
            else:
                dynamic_dict['price_range'].add(new_word)
                categories.append((new_word, 'price_range'))
                categorized_words.add(new_word)

    # Levenshtein distance for uncategorized words
    categories_list = [('food_type', basic_dict['food_type'], dynamic_dict['food_type']),
                       ('price_range', basic_dict['price_range'], dynamic_dict['price_range']),
                       ('location', basic_dict['location'], dynamic_dict['location'])]

    for word in sentence_words:
        if word not in categorized_words and word not in stopwords and len(word) >= 3:
            for category, basic_words, dynamic_words in categories_list:
                closest_match = apply_levenshtein(word, basic_words) or apply_levenshtein(word, dynamic_words)
                if closest_match:
                    categories.append((closest_match, category))
                    dynamic_words.add(closest_match)
                    break

    return categories

def update_dict(sentence, basic_dict, dynamic_dict):
    categories = categorize_words(sentence, basic_dict, dynamic_dict)
    sentence_words = sentence.lower().split()
    filtered_words = [word for word in sentence_words if word not in stopwords]
    word_counter.update(filtered_words)
    return categories

# use the categorization and print the results
for sentence in blackboard_sentences:
    categories = update_dict(sentence, basic_dict, dynamic_dict)
    print(f"Sentence: {sentence}")
    print(f"Categorized: {categories}\n")

# Print the table of most commonly used words
def most_common_words(counter, top_n=30):
    common_words = [(word, freq) for word, freq in counter.most_common() if len(word) >= 3 and word not in stopwords]
    common_words = common_words[:top_n]
    for word, freq in common_words:
        print(f"{word}: {freq}")

print("\nMost common words for this data excluding words < 3 characters and basic noise words:")
most_common_words(word_counter)

# what was added to the dictionary 
print("\ndynamic dictionary:")
print(dynamic_dict)