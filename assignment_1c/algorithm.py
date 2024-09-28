# algorithm.py

from collections import Counter
from collections import defaultdict


class TextProcessor:
    def __init__(self):
        self.basic_dict = {
            'food_type': ['asian', 'korean', 'japanese', 'indian', 'thai', 'cuban', 'french',
                          'mexican', 'western', 'catalan', 'mediterranean', 'seafood', 'oriental',
                          'scottish', 'austrian', 'international', 'eritrean', 'spanish',
                          'australian', 'turkish'],
            'price_range': ['moderate', 'moderately', 'expensive', 'cheap'],
            'location': ['north', 'south', 'east', 'west', 'center', 'any']
        }

        self.dynamic_dict = {
            'food_type': set(),
            'price_range': set(),
            'location': set()
        }

        self.stopwords = set(['a', 'an', 'the', 'in', 'that', 'priced', 'would'])
        self.word_counter = Counter()

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

    def apply_levenshtein(self, word, category_words, threshold=1):
        closest_word = None
        min_distance = float('inf')
        for new_word in category_words:
            distance = self.levenshtein_recursive(word, new_word, len(word), len(new_word))
            if distance < min_distance and distance <= threshold:
                closest_word = new_word
                min_distance = distance
        return closest_word

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
                elif location_word in self.dynamic_dict['location']:
                    categories.append((location_word, 'location'))
                    categorized_words.add(location_word)

        # Check for "food" pattern
        for i in range(1, len(sentence_words)):
            if sentence_words[i] == 'food' and sentence_words[i - 1] not in self.stopwords:
                food_type_word = sentence_words[i - 1]
                if food_type_word in self.basic_dict['food_type']:
                    categories.append((food_type_word, 'food_type'))
                    categorized_words.add(food_type_word)
                elif food_type_word in self.dynamic_dict['food_type']:
                    categories.append((food_type_word, 'food_type'))
                    categorized_words.add(food_type_word)
                else:
                    self.dynamic_dict['food_type'].add(food_type_word)
                    categories.append((food_type_word, 'food_type'))
                    categorized_words.add(food_type_word)

        # Check for "priced" pattern
        for i in range(2, len(sentence_words)):
            if sentence_words[i] == 'priced' and sentence_words[i - 1] not in self.stopwords:
                price_word = sentence_words[i - 1]
                if price_word in self.basic_dict['price_range']:
                    categories.append((price_word, 'price_range'))
                    categorized_words.add(price_word)
                elif price_word in self.dynamic_dict['price_range']:
                    categories.append((price_word, 'price_range'))
                    categorized_words.add(price_word)
                else:
                    self.dynamic_dict['price_range'].add(price_word)
                    categories.append((price_word, 'price_range'))
                    categorized_words.add(price_word)

        # Levenshtein distance for uncategorized words
        categories_list = [
            ('food_type', self.basic_dict['food_type'], self.dynamic_dict['food_type']),
            ('price_range', self.basic_dict['price_range'], self.dynamic_dict['price_range']),
            ('location', self.basic_dict['location'], self.dynamic_dict['location'])
        ]

        for word in sentence_words:
            if word not in categorized_words and word not in self.stopwords and len(word) >= 3:
                for category, basic_words, dynamic_words in categories_list:
                    closest_match = self.apply_levenshtein(word, basic_words) or self.apply_levenshtein(word,
                                                                                                        dynamic_words)
                    if closest_match:
                        categories.append((closest_match, category))
                        dynamic_words.add(closest_match)
                        break

        return categories
