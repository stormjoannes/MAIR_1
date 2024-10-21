from assignment_1a.DecisionTreeClassifier import vectorizer, clf_tree
from algorithm import TextProcessor
from restaurant_selector import RestaurantSelector
import random
import time


class DialogManager:
    def __init__(self, amount_of_recommendations, response_delay, language_style, transparent, memory):
        self.state = "welcome"
        self.formality = "formal"
        self.preferences = {
            "location": None,
            "food_type": None,
            "price_range": None,
            "romantic": None,
            "children": None,
            'touristic': None,
            'assigned_seats': None,
        }
        self.response = True
        self.restaurant = None
        self.amount_of_recommendations = amount_of_recommendations
        self.text_processor = TextProcessor()
        self.restaurant_selector = RestaurantSelector()
        self.changes_counter = 0
        self.preferences_name = list(self.preferences.keys())
        self.rules = {
            1: {'antecedents': [('price_range', 'cheap'), ('food_quality', 'good')], 'consequent': ('touristic', True), 'description': 'A cheap restaurant with good food attracts tourists'},
            2: {'antecedents': [('cuisine', 'Romanian')], 'consequent': ('touristic', False), 'description': 'Romanian cuisine is unknown for most tourists and they prefer familiar food'},
            3: {'antecedents': [('crowdedness', 'busy')], 'consequent': ('assigned_seats', True), 'description': 'In a busy restaurant the waiter decides where you sit'},
            4: {'antecedents': [('stay_duration', 'long')], 'consequent': ('suitable_for_children', False), 'description': 'Spending a long time is not advised when taking children'},
            5: {'antecedents': [('crowdedness', 'busy')], 'consequent': ('romantic', False), 'description': 'A busy restaurant is not romantic'},
            6: {'antecedents': [('stay_duration', 'long')], 'consequent': ('romantic', True), 'description': 'Spending a long time in a restaurant is romantic'},
        }
        # Variables for anthropomorphic system
        self.response_delay = response_delay
        self.language_style = language_style
        self.transparent = transparent
        self.memory = memory

    def println(self, output):
        """Prints the output with a typing effect for delay of response."""
        typing_text = 'typing...'
        print(typing_text, end='', flush=True)
        time.sleep(self.response_delay)  # Wait for 2 seconds
        print('\r' + f"System: {output}", end='', flush=True)
        print()  # Move to the next line

    def classify_dialog_act(self, user_utterance):
        """Classify the dialog act of the user utterance using the Decision Tree classifier."""
        input_bow = vectorizer.transform([user_utterance])
        prediction = clf_tree.predict(input_bow)
        return prediction[0]

    def get_response(self):
        """Returns the response based on the current state and formality level."""
        responses = {
            "welcome": {
                "formal": "Welcome! How may I assist you today?",
                "informal": "Hey there! What can I do for ya?"
            },
            "goodbye": {
                "formal": "Thank you for using our service. Goodbye!",
                "informal": "Alright, take care!"
            }
        }
        return responses[self.state][self.formality]

    def reset_dialog(self):
        """Resets the dialog state and preferences to the initial state."""
        self.state = "welcome"
        self.preferences = {key: None for key in self.preferences}
        self.formality = "formal"
        self.println("The dialog has been reset. How may I assist you?")

    def ask_additional_requirements(self):
        """Asks the user for additional requirements like romantic setting or a place suitable for children."""
        if self.preferences.get('romantic') is not None or self.preferences.get('children') is not None:
            self.println("Got it. Let's find the perfect spot for you!")
            self.state = "make_recommendation"
        else:
            self.println("Do you have any specific requirements like needing a romantic setting or a place suitable for children?")

    def extract_additional_preferences(self, user_input):
        """Extracts the additional preferences like romantic setting or a place suitable for children."""
        preferences_map = {
            'romantic': 'romantic',
            'children': 'children',
            'touristic': 'touristic',
            'assigned': 'assigned_seats'
        }

        for key, preference in preferences_map.items():
            if key in user_input:
                self.preferences[preference] = 'yes' in user_input or key in user_input

        self.response = False
        self.state = "make_recommendation"

    def redirection(self, category):
        """Redirects the user to the next step based on the category of preference."""
        if self.state != "welcome":
            self.response = False

        match category:
            case "food_type":
                if self.preferences["location"] is None:
                    self.state = "ask_location"
                elif self.preferences["price_range"] is None:
                    self.state = "ask_price_range"
                else:
                    self.state = "ask_specific_requirements"
            case "location":
                if self.preferences["food_type"] is None:
                    self.state = "ask_food_type"
                elif self.preferences["price_range"] is None:
                    self.state = "ask_price_range"
                else:
                    self.state = "ask_specific_requirements"
            case "price_range":
                if self.preferences["location"] is None:
                    self.state = "ask_location"
                elif self.preferences["food_type"] is None:
                    self.state = "ask_food_type"
                else:
                    self.state = "ask_specific_requirements"
            case _:
                self.state = "ask_specific_requirements"

        if self.preferences_ready():
            self.ask_additional_requirements()
            self.response = True

    def extract_preferences(self, user_utterance, input_category=None):
        """Extracts the preferences from the user utterance based on the category of preference."""
        categories = self.text_processor.categorize_words(user_utterance)

        for word in user_utterance.lower().split():
            if word == 'dontcare':
                self.preferences[input_category] = 'blank'  # Set to None to ignore this preference

        for word, category in categories:
            if category in self.preferences and not self.preferences[category]:
                self.preferences[category] = word

    def preferences_ready(self):
        return all(self.preferences[key] is not None for key in ['location', 'food_type', 'price_range'])

    def make_recommendation(self):
        """Makes a recommendation based on the user preferences and inferred properties."""

        filtered_restaurants = self.restaurant_selector.recommend_restaurant(
            self.preferences['food_type'],
            self.preferences['price_range'],
            self.preferences['location'],
            self.preferences  # Pass the inferred properties
        )
        if isinstance(filtered_restaurants, str):
            self.println(filtered_restaurants)
            self.println("Do you want to modify any preference in order to keep searching?")
            self.state = "no_match"
            return

        recommendations = {}
        for i in range(self.amount_of_recommendations if len(filtered_restaurants) > self.amount_of_recommendations
                       else len(filtered_restaurants)):
            # Select a random restaurant for recommendation
            recommendation = filtered_restaurants.sample(n=1).iloc[0]
            recommendations[i + 1] = recommendation
            filtered_restaurants.drop(recommendation.name, inplace=True)
            self.println(f"Recommendation {i + 1}: '{recommendation['restaurantname']}', a(n) {recommendation['pricerange']} {recommendation['food']} restaurant in the {recommendation['area']}.")

        self.recommendation_selector(recommendations)

        self.state = "request_further_details"

    def recommendation_selector(self, recommendations):
        """Selects a restaurant from the recommendations based on user input."""
        if len(recommendations) > 1:
            self.println(f"Which restaurant would you like more information about {tuple(recommendations.keys())}?")

            while True:
                user_input = int(input("You: "))
                if user_input in recommendations.keys():
                    self.restaurant = recommendations[user_input]
                    self.println(f"You selected '{self.restaurant['restaurantname']}'. Do you want the phone number?")
                    break
                else:
                    self.println("Please select a valid recommendation number.")
        else:
            self.restaurant = recommendations

    def handle_state(self, dialog_act, user_utterance):
        """Handles the dialog state based on the dialog act and user utterance."""
        if self.state == "welcome":
            if dialog_act == "hello":
                self.println("Welcome! Please, tell me the location where you want to find a restaurant.")
                self.state = "ask_location"
            elif dialog_act == "inform":
                self.extract_preferences(user_utterance)
                self.response = False
                non_none_preferences = [key for key, value in self.preferences.items() if value is not None]
                if len(non_none_preferences) > 0:
                    self.redirection(non_none_preferences[0])
                else:
                    self.state = "ask_location"
            elif dialog_act == "restart":
                self.reset_dialog()
            elif dialog_act in ["bye", "negate"]:
                self.state = "goodbye"
                self.println(self.get_response())
            else:
                self.println("Sorry, I didn't understand. Could you tell me your preferences again?")

        elif self.state == "start_over":
            self.reset_dialog()

        elif self.state == "ask_location":
            if dialog_act == "inform":
                self.extract_preferences(user_utterance, 'location')
                if self.preferences["location"]:
                    if self.preferences["location"] == 'blank':
                        self.println("Understood, you don't have a specific location in mind for the restaurant.")
                    else:
                        self.println(f"Got it, you're looking for a restaurant in {self.preferences['location']}.")
                    self.redirection("location")
                else:
                    self.println("Could you please tell me the location you want to find a restaurant?")
            elif dialog_act in ["bye", "negate"]:
                self.state = "goodbye"
                self.println(self.get_response())

        elif self.state == "ask_food_type":
            if dialog_act == "inform":
                self.extract_preferences(user_utterance, 'food_type')
                if self.preferences["food_type"]:
                    if self.preferences["food_type"] == 'blank':
                        self.println("Understood, you don't have a specific food type in mind.")
                    else:
                        self.println(f"Great, you're looking for {self.preferences['food_type']} food.")
                    self.redirection("food_type")
                else:
                    self.println("Could you please tell me the type of food you prefer?")
            elif dialog_act in ["bye", "negate"]:
                self.state = "goodbye"
                self.println(self.get_response())

        elif self.state == "ask_price_range":
            if dialog_act == "inform":
                self.extract_preferences(user_utterance, 'price_range')
                if self.preferences["price_range"]:
                    if self.preferences["price_range"] == 'blank':
                        self.println("Understood, you don't have a specific price range in mind.")
                    else:
                        self.println(f"You're looking for a(n) {self.preferences['price_range']} restaurant.")
                    self.redirection("price_range")
                else:
                    self.println("Could you please tell me your price range (choose from cheap, moderate, expensive)?")
            elif dialog_act in ["bye", "negate"]:
                self.state = "goodbye"
                self.println(self.get_response())

        elif self.state == "ask_specific_requirements":
            if dialog_act == "negate":
                self.response = False
                self.state = "make_recommendation"
            elif dialog_act == "affirm":
                self.println("Please specify if you need a romantic setting, touristic restaurant, assigned seats or a place suitable for children.")
            else:
                self.extract_additional_preferences(user_utterance)
                self.response = False
                self.state = "make_recommendation"

        elif self.state == "make_recommendation":
            self.make_recommendation()

        elif self.state == "no_match":
            if dialog_act == "negate":
                self.println("Sorry, have a great day! Goodbye.")
                self.state = "change_preferences"
            else:
                self.response = False
                self.state = "changes"

        elif self.state == "changes":
            if dialog_act == "inform":
                self.changes_counter -= 1
            elif dialog_act == "negate":
                self.println("Okay!")
            else:
                self.preferences[self.preferences_name[self.changes_counter]] = None
            self.changes_counter += 1

            if self.changes_counter == 3:
                self.changes_counter = 0
                self.response = False
                self.state = "welcome"
            else:
                self.println(f"Do you want to change the {self.preferences_name[self.changes_counter]} of the restaurant? (Yes/No)")
                self.state = "changes"

        elif self.state == "request_further_details":
            if dialog_act == "negate":
                self.println("Okay, have a great day! Goodbye.")
                self.state = "goodbye"
            else:
                self.println(f"The phone number is {self.restaurant['phone']}. Do you want the address?")
                self.state = "provide_address"

        elif self.state == "provide_address":
            if dialog_act == "negate":
                self.println("Alright. Thank you, goodbye!")
                self.state = "goodbye"
            else:
                self.println(f"The address is {self.restaurant['addr']}. Would you like the postal code?")
                self.state = "provide_postalcode"

        elif self.state == "provide_postalcode":
            if dialog_act == "negate":
                self.println("Okay, have a nice day!")
                self.state = "goodbye"
            else:
                self.println(f"The postal code is {self.restaurant['postcode']}. Thank you for using the system!")
                self.state = "goodbye"

    def next_state(self, user_utterance):
        """Determines the next state based on the dialog act and user utterance."""
        dialog_act = self.classify_dialog_act(user_utterance)
        self.handle_state(dialog_act, user_utterance)

    def run(self):
        """Runs the dialog system."""
        # Pick random formality level, formal or informal
        self.formality = random.choice(["formal", "informal"])

        # Welcome the user
        self.println(self.get_response())

        while self.state != "goodbye":
            if self.response:
                user_input = input("You: ").lower()
                self.next_state(user_input)
            else:
                user_input = ""
                self.response = True
                self.next_state(user_input)
