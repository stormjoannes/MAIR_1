from assignment_1a.DTC import vectorizer, clf_tree
from algorithm import TextProcessor
from restaurant_selector import RestaurantSelector

class DialogManager:
    def __init__(self):
        self.state = "welcome"
        self.formality = "formal"
        self.preferences = {
            "location": None,
            "food_type": None,
            "price_range": None,
            "romantic": None,
            "children": None
        }
        self.response = True
        self.restaurant = None
        self.other_options = None
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

    def apply_rules(self):
        properties = {}  # Track properties like 'romantic', 'touristic', etc.
        for rule_id, rule in self.rules.items():
            if all(self.preferences.get(ant[0]) == ant[1] for ant in rule['antecedents']):
                consequent, value = rule['consequent']
                if consequent in properties:
                    # Handling contradictions
                    if properties[consequent] != value:
                        print(f"Contradiction detected for {consequent}. Resolving based on user input or additional logic.")
                        # Resolution logic could go here
                        continue
                properties[consequent] = value
        return properties

    def classify_dialog_act(self, user_utterance):
        input_bow = vectorizer.transform([user_utterance])
        prediction = clf_tree.predict(input_bow)
        return prediction[0]

    def get_response(self, message_key):
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
        return responses[message_key][self.formality]

    def reset_dialog(self):
        self.state = "welcome"
        self.preferences = {key: None for key in self.preferences}
        self.formality = "formal"
        print("System: The dialog has been reset. How may I assist you?")

    def ask_additional_requirements(self):
        if self.preferences.get('romantic') is not None or self.preferences.get('children') is not None:
            print("System: Got it. Let's find the perfect spot for you!")
            self.state = "make_recommendation"
        else:
            print(
                "System: Do you have any specific requirements like needing a romantic setting or a place suitable for children?")

    def extract_additional_preferences(self, user_input):
        if 'romantic' in user_input or 'children' in user_input:
            if 'romantic' in user_input:
                self.preferences['romantic'] = True if 'yes' in user_input or 'romantic' in user_input else False
            if 'children' in user_input:
                self.preferences['children'] = True if 'yes' in user_input or 'children' in user_input else False
            self.state = "make_recommendation"
        else:
            self.state = "ask_additional_requirements"  # Ensure correct state transition

    def redirection(self, category):
        if self.state != "welcome":
            self.response = False

        if category == "food_type":
            if self.preferences["location"] is None:
                self.state = "ask_location"
            elif self.preferences["price_range"] is None:
                self.state = "ask_price_range"
            else:
                self.state = "make_recommendation"
        elif category == "location":
            if self.preferences["food_type"] is None:
                self.state = "ask_food_type"
            elif self.preferences["price_range"] is None:
                self.state = "ask_price_range"
            else:
                self.state = "make_recommendation"
        else:  # price_range
            if self.preferences["location"] is None:
                self.state = "ask_location"
            elif self.preferences["food_type"] is None:
                self.state = "ask_food_type"
            else:
                self.state = "make_recommendation"

    def extract_preferences(self, user_utterance, input_category=None):
        categories = self.text_processor.categorize_words(user_utterance)

        print("User utterance: ", user_utterance)
        print("Categories: ", categories)

        for word in user_utterance.lower().split():
            if word == 'dontcare':
                print("SUCCES")
                print(self.preferences)
                self.preferences[input_category] = 'blank'  # Set to None to ignore this preference

        for word, category in categories:
            if category in self.preferences and not self.preferences[category]:
                self.preferences[category] = word
        if self.preferences_ready():
            self.ask_additional_requirements()

    def preferences_ready(self):
        return all(self.preferences[key] is not None for key in ['location', 'food_type', 'price_range'])

    def make_recommendation(self):
        # Applying rules to determine properties like 'romantic' and 'children'
        properties = self.apply_rules()

        # Making recommendation based on preferences and inferred properties
        recommendation, remaining_options = self.restaurant_selector.recommend_restaurant(
            self.preferences['food_type'],
            self.preferences['price_range'],
            self.preferences['location'],
            properties  # Pass the inferred properties
        )
        if isinstance(recommendation, str):
            print(recommendation)
            print("System: Do you want to modify any preference in order to keep searching?")
            self.state = "no_match"
            return

        self.restaurant = recommendation
        self.other_options = remaining_options
        print(
            f"System: Recommending '{recommendation['restaurantname']}', a(n) {recommendation['pricerange']} {recommendation['food']} restaurant in the {recommendation['area']}.")
        self.state = "request_further_details"

    def handle_state(self, dialog_act, user_utterance):
        if self.state == "welcome":
            if dialog_act == "hello":
                print("System: Welcome! Please, tell me the location where you want to find a restaurant.")
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
                print("System: Goodbye!")
            else:
                print("System: Sorry, I didn't understand. Could you tell me your preferences again?")

        elif self.state == "start_over":
            self.reset_dialog()

        elif self.state == "ask_location":
            if dialog_act == "inform":
                self.extract_preferences(user_utterance, 'location')
                if self.preferences["location"]:
                    if self.preferences["location"] == 'blank':
                        print("System: Understood, you don't have a specific location in mind for the restaurant.")
                    else:
                        print(f"System: Got it, you're looking for a restaurant in {self.preferences['location']}.")
                    self.redirection("location")
                else:
                    print("System: Could you please tell me the location you want to find a restaurant?")
            elif dialog_act in ["bye", "negate"]:
                self.state = "goodbye"
                print("System: Goodbye!")

        elif self.state == "ask_food_type":
            if dialog_act == "inform":
                self.extract_preferences(user_utterance, 'food_type')
                if self.preferences["food_type"]:
                    if self.preferences["food_type"] == 'blank':
                        print("System: Understood, you don't have a specific food type in mind.")
                    else:
                        print(f"System: Great, you're looking for {self.preferences['food_type']} food.")
                    self.redirection("food_type")
                else:
                    print("System: Could you please tell me the type of food you prefer?")
            elif dialog_act in ["bye", "negate"]:
                self.state = "goodbye"
                print("System: Goodbye!")

        elif self.state == "ask_price_range":
            if dialog_act == "inform":
                self.extract_preferences(user_utterance, 'price_range')
                if self.preferences["price_range"]:
                    if self.preferences["price_range"] == 'blank':
                        print("System: Understood, you don't have a specific price range in mind.")
                    else:
                        print(f"System: You're looking for a {self.preferences['price_range']} restaurant.")
                    self.state = "ask_specific_requirements"
                else:
                    print("System: Could you please tell me your price range?")
            elif dialog_act in ["bye", "negate"]:
                self.state = "goodbye"
                print("System: Goodbye!")

        elif self.state == "ask_specific_requirements":
            self.extract_additional_preferences(user_utterance)
            if self.preferences_ready():
                self.state = "make_recommendation"
            else:
                print("System: Please specify if you need a romantic setting or a place suitable for children.")

        elif self.state == "make_recommendation":
            self.make_recommendation()

        elif self.state == "no_match":
            if dialog_act == "negate":
                print("System: Sorry, have a great day! Goodbye.")
                self.state = "change_preferences"
            else:
                self.response = False
                self.state = "changes"

        elif self.state == "changes":
            if dialog_act == "inform":
                self.changes_counter -= 1
            elif dialog_act == "negate":
                print("System: Okay!")
            else:
                self.preferences[self.preferences_name[self.changes_counter]] = None
            self.changes_counter += 1

            if self.changes_counter == 3:
                self.changes_counter = 0
                self.response = False
                self.state = "welcome"
            else:
                print(f"System: Do you want to change the {self.preferences_name[self.changes_counter]} of the restaurant? (Yes/No)")
                self.state = "changes"

        elif self.state == "request_further_details":
            if dialog_act == "negate":
                print("System: Okay, have a great day! Goodbye.")
                self.state = "goodbye"
            else:
                print(f"System: The phone number is {self.restaurant['phone']}. Do you want the address?")
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

    def next_state(self, user_utterance):
        dialog_act = self.classify_dialog_act(user_utterance)
        self.handle_state(dialog_act, user_utterance)

    def run(self):
        print("System: Hello! How can I help you today?")
        while self.state != "goodbye":
            if self.response:
                user_input = input("You: ").lower()
                self.next_state(user_input)
            else:
                user_input = ""
                self.response = True
                self.next_state(user_input)
