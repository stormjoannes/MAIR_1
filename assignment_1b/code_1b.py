# TASK 4

import pandas as pd
import random


# Load csv of restaurants
restaurants_df = pd.read_csv('/content/restaurant_info.csv')


# Updated filtering function to match the new CSV fields
def filter_restaurants(food_type=None, price_range=None, area=None):
   # Create a filtered DataFrame based on the criteria
   filtered_df = restaurants_df


   # Filter by food type if specified
   if food_type:
       filtered_df = filtered_df[filtered_df['food'].str.contains(food_type, case=False, na=False)]


   # Filter by price range if specified
   if price_range:
       filtered_df = filtered_df[filtered_df['pricerange'].str.contains(price_range, case=False, na=False)]


   # Filter by area if specified
   if area:
       filtered_df = filtered_df[filtered_df['area'].str.contains(area, case=False, na=False)]


   return filtered_df




# Function to recommend a restaurant and store remaining options
def recommend_restaurant(food_type=None, price_range=None, area=None):
   # Filter restaurants based on user preferences
   filtered_restaurants = filter_restaurants(food_type, price_range, area)


   # Check if any restaurants match the criteria
   if filtered_restaurants.empty:
       return "Sorry, no restaurant matches your preferences."


   # Randomly select one restaurant as recommendation
   recommendation = filtered_restaurants.sample(n=1).iloc[0]


   # Store the remaining restaurants after the recommendation
   remaining_restaurants = filtered_restaurants.drop(recommendation.name)


   return recommendation, remaining_restaurants

# TASK 3: ALGORITHM TO GET/COLLECT INFORMATION 
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
    'food_type': ['asian', 'korean', 'japanese', 'indian', 'thai', 'cuban', 'french', 'mexican', 'western','catalan','mediterranean', 'seafood', 'asian', 'oriental', 'scottish', 'austrian', 'international', 'eirtrean', 'spanish', 'australian', 'turkish'],
    'price_range': ['moderate','moderately', 'expensive', 'cheap'],
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

# --------------------------------------------TASK 2---------------------------------------

# State transition function for the dialog system
class DialogManager:
    def __init__(self):
        self.state = "welcome"  # Initial state
        self.preferences = {
            "area": None,
            "food_type": None,
            "price_range": None
        }
        self.response = True
        self.restaurant = None
        self.other_options = None

    def classify_dialog_act(self, user_utterance):
        input_bow = vectorizer.transform([user_utterance])
        # ----------------------MODEL SELECTION-------------------
        prediction = clf_tree.predict(input_bow)
        # print(prediction)
        return prediction

    def redirection(self, category):

      if self.state != "welcome":
        self.response = False

      if category == "food_type":
        if self.preferences["area"] == None:
          self.state = "ask_area"
        elif self.preferences["price_range"] == None:
          self.state = "ask_price_range"
        else:
          self.state = "make_recommendation"
      elif category == "area":
        if self.preferences["food_type"] == None:
          self.state = "ask_food_type"
        elif self.preferences["price_range"] == None:
          self.state = "ask_price_range"
        else:
          self.state = "make_recommendation"
      else:
        if self.preferences["area"] == None:
          self.state = "ask_area"
        elif self.preferences["food_type"] == None:
          self.state = "ask_food_type"
        else:
          self.state = "make_recommendation"



    def extract_preferences(self, user_utterance):
        """
        This function extracts preferences using pattern matching on utterances classified as 'inform', using
        Levenshtein edit distance if necessary. It incorporates the previous functions to match the user's preferences.
        """
        categories = categorize_words(user_utterance, basic_dict, dynamic_dict)
        for word, category in categories:
            if category == 'food_type' and not self.preferences["food_type"]:
                self.preferences["food_type"] = word
                self.redirection(category)
            elif category == 'price_range' and not self.preferences["price_range"]:
                self.preferences["price_range"] = word
                self.redirection(category)
            elif category == 'location' and not self.preferences["area"]:
                self.preferences["area"] = word
                self.redirection(category)
        # print(f"Updated preferences: {self.preferences}")

    def next_state(self, user_utterance):
        # Classify the dialog act using the model
        dialog_act = self.classify_dialog_act(user_utterance)

        # State transition
        if self.state == "welcome":
            if dialog_act == "hello":
                print("System: Welcome! Please, tell me the area where you want to find a restaurant.")
                self.state = "ask_food_type"
            elif dialog_act == "inform":
                self.extract_preferences(user_utterance)
                self.response = False
                self.state = "ask_area" if not self.preferences["area"] else "ask_food_type"
            elif dialog_act == "restart":
                self.state = "start_over"
            elif dialog_act == "bye" or "negate":
                self.state = "goodbye"
            else:
                print("System: Sorry, I didn't understand. Could you tell me your preferences again?")
        
        elif self.state == "start_over":
          print("System: We start over!")
          self.preferences["area"] = None
          self.preferences["food_type"] = None
          self.preferences["price_range"] = None
          self.state == "welcome"

        elif self.state == "ask_preferences":
            print("System: Could you please tell me the area you want to find a restaurant?")
            if dialog_act == "inform":
                self.extract_preferences(user_utterance)
                self.state = "ask_food_type"
            elif dialog_act == "bye" or "negate":
                self.state = "welcome"

        elif self.state == "ask_area":
            if dialog_act == "inform":
                self.extract_preferences(user_utterance)
                if self.preferences["area"]:
                    print(f"System: Got it, you're looking for a restaurant in {self.preferences['area']}.")
                    self.redirection("area")
                else:
                    print("System: Could you please tell me the area you want to find a restaurant?")

        elif self.state == "ask_food_type":
            if dialog_act == "inform":
                self.extract_preferences(user_utterance)
                if self.preferences["food_type"]:
                    print(f"System: Great, you're looking for {self.preferences['food_type']} food. What's your price range?")
                    self.state = "ask_price_range"
                else:
                    print("System: Could you please tell me the type of food you prefer?")

        elif self.state == "ask_price_range":
            if dialog_act == "inform":
                self.extract_preferences(user_utterance)
                if self.preferences["price_range"]:
                    print(f"System: You're looking for a {self.preferences['price_range']} restaurant. I'll find the best options for you!")
                    self.state = "make_recommendation"
                else:
                    print("System: Could you please tell me your price range?")

        elif self.state == "make_recommendation":
            try:
              recommended_restaurant, remaining_options = recommend_restaurant(self.preferences['food_type'], self.preferences['price_range'], self.preferences['area'])
              self.restaurant = recommended_restaurant
              self.other_options = remaining_options
              print(f"System: Recommending a {self.preferences['price_range']} {self.preferences['food_type']} restaurant in {self.preferences['area']}: {recommended_restaurant['restaurantname']}. Do you want the phone number?")
              self.state = "request_further_details"
            except ValueError:
              print("Sorry, no restaurant matches your preferences. Let's start again!")
              self.response = False
              self.state = "welcome"


        elif self.state == "request_further_details":
            if dialog_act == "negate":
                print("System: Okay, have a great day! Goodbye.")
                self.state = "goodbye"
            else:
                print(f"System: The phone number is {self.restaurant['phone']} . Do you want the address?")
                self.state = "provide_address"

        elif self.state == "provide_address":
            if dialog_act == "negate":
                print("System: Alright. Thank you, goodbye!")
                self.state = "goodbye"
            else:
                print(f"System: The address is {self.restaurant['addr']}. Would you like the postal code?")
                self.state = "provide_postalcode"

        elif self.state == "provide_postalcode":
            if dialog_act == "negate":
                print("System: Okay, have a nice day!")
                self.state = "goodbye"
            else:
                print(f"System: The postal code is {self.restaurant['postcode']}. Thank you for using the system!")
                self.state = "goodbye"

        elif self.state == "goodbye":
            print("System: Goodbye!")

    def run(self):
        # Start
        print("System: Hello! How can I help you today?")
        while self.state != "goodbye":
            if self.response:
              user_input = input("You: ").lower()
              self.next_state(user_input)
            else:
              user_input = ""
              self.response = True
              self.next_state(user_input)
       
dialog_manager = DialogManager()
dialog_manager.run()