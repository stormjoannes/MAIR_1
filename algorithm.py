class Algorithm:
    def __init__(self, basic_dict, dynamic_dict):
        self.basic_dict = basic_dict
        self.dynamic_dict = dynamic_dict
        self.stopwords = {'a', 'an', 'the', 'in', 'that', 'priced', 'would'}

    # Levenshtein distance function
    def levenshtein_recursive(self, str1, str2, m, n):
        if m == 0:
            return n
        if n == 0:
            return m
        if str1[m - 1] == str2[n - 1]:
            return self.levenshtein_recursive(str1, str2, m - 1, n - 1)
        return 1 + min(
            self.levenshtein_recursive(str1, str2, m, n - 1),  # Insert
            self.levenshtein_recursive(str1, str2, m - 1, n),  # Remove
            self.levenshtein_recursive(str1, str2, m - 1, n - 1)  # Replace
        )

    # Find closest match
    def apply_levenshtein(self, word, category_words, threshold=1):
        closest_word = None
        min_distance = float('inf')
        for new_word in category_words:
            distance = self.levenshtein_recursive(word, new_word, len(word), len(new_word))
            if distance < min_distance and distance <= threshold:
                closest_word = new_word
                min_distance = distance
        return closest_word

    # Categorize words
    def categorize_words(self, sentence):
        sentence_words = sentence.lower().split()
        categories = []
        categorized_words = set()

        # Check for "in the" location pattern
        for i in range(1, len(sentence_words) - 1):
            if sentence_words[i - 1:i + 1] == ['in', 'the']:
                location_word = sentence_words[i + 1]
                if location_word in self.basic_dict['location']:
                    categories.append((location_word, 'location'))
                    categorized_words.add(location_word)

        # Check for food type and price patterns
        for i in range(len(sentence_words)):
            if sentence_words[i] == 'food':
                food_type_word = sentence_words[i - 1]
                if food_type_word not in self.stopwords:
                    categories.append((food_type_word, 'food_type'))

            if sentence_words[i] == 'priced':
                price_word = sentence_words[i - 1]
                if price_word not in self.stopwords:
                    categories.append((price_word, 'price_range'))

        # Levenshtein distance for uncategorized words
        for word in sentence_words:
            if word not in categorized_words and word not in self.stopwords:
                for category, basic_words, dynamic_words in [
                    ('food_type', self.basic_dict['food_type'], self.dynamic_dict['food_type']),
                    ('price_range', self.basic_dict['price_range'], self.dynamic_dict['price_range']),
                    ('location', self.basic_dict['location'], self.dynamic_dict['location'])
                ]:
                    closest_match = self.apply_levenshtein(word, basic_words) or self.apply_levenshtein(word, dynamic_words)
                    if closest_match:
                        categories.append((closest_match, category))
                        dynamic_words.add(closest_match)
                        break

        return categories
