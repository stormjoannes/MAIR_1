# dialog_system.py

from assignment_1a.DTC import vectorizer, clf_tree
from algorithm import TextProcessor
from restaurant_selector import RestaurantSelector


class DialogManager:
    def __init__(self):
        self.state = "welcome"
        self.preferences = {
            "area": None,
            "food_type": None,
            "price_range": None
        }
        self.response = True
        self.restaurant = None
        self.other_options = None
        self.text_processor = TextProcessor()
        self.restaurant_selector = RestaurantSelector()
        self.changes_counter = 0
        self.preferences_name = list(self.preferences.keys())

    def classify_dialog_act(self, user_utterance):
        input_bow = vectorizer.transform([user_utterance])
        prediction = clf_tree.predict(input_bow)
        return prediction[0]

    def redirection(self, category):
        if self.state != "welcome":
            self.response = False

        if category == "food_type":
            if self.preferences["area"] is None:
                self.state = "ask_area"
            elif self.preferences["price_range"] is None:
                self.state = "ask_price_range"
            else:
                self.state = "make_recommendation"
        elif category == "area":
            if self.preferences["food_type"] is None:
                self.state = "ask_food_type"
            elif self.preferences["price_range"] is None:
                self.state = "ask_price_range"
            else:
                self.state = "make_recommendation"
        else:  # price_range
            if self.preferences["area"] is None:
                self.state = "ask_area"
            elif self.preferences["food_type"] is None:
                self.state = "ask_food_type"
            else:
                self.state = "make_recommendation"

    def extract_preferences(self, user_utterance):
        categories = self.text_processor.categorize_words(user_utterance)
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

    def make_recommendation(self):
        recommendation, remaining_options = self.restaurant_selector.recommend_restaurant(
            self.preferences['food_type'],
            self.preferences['price_range'],
            self.preferences['area']
        )
        if isinstance(recommendation, str):
            print(recommendation)
            print("System: Do you want to modify any preference in order to keep searching?")
            self.state = "no_match"
            return

        self.restaurant = recommendation
        self.other_options = remaining_options
        print(f"System: Recommending a {self.preferences['price_range']} {self.preferences['food_type']} restaurant in {self.preferences['area']}: {recommendation['restaurantname']}. Do you want the phone number?")
        self.state = "request_further_details"

    def handle_state(self, dialog_act, user_utterance):

        if self.state == "welcome":
            if dialog_act == "hello":
                print("System: Welcome! Please, tell me the area where you want to find a restaurant.")
                self.state = "ask_area"
            elif dialog_act == "inform":
                self.extract_preferences(user_utterance)
                self.response = False
                non_none_preferences = [key for key, value in self.preferences.items() if value is not None]
                if len(non_none_preferences) > 0:
                    self.redirection(non_none_preferences[0])
                else:
                    self.state = "ask_area"
            elif dialog_act == "restart":
                self.state = "start_over"
            elif dialog_act in ["bye", "negate"]:
                self.state = "goodbye"
                print("System: Goodbye!")
            else:
                print("System: Sorry, I didn't understand. Could you tell me your preferences again?")

        elif self.state == "start_over":
            print("System: We start over!")
            self.response = False
            self.preferences = {key: None for key in self.preferences}
            self.state = "ask_area"

        elif self.state == "ask_area":
            if dialog_act == "inform":
                self.extract_preferences(user_utterance)
                if self.preferences["area"]:
                    print(f"System: Got it, you're looking for a restaurant in {self.preferences['area']}.")
                    self.redirection("area")
                else:
                    print("System: Could you please tell me the area you want to find a restaurant?")
            elif dialog_act in ["bye", "negate"]:
                self.state = "goodbye"
                print("System: Goodbye!")

        elif self.state == "ask_food_type":
            if dialog_act == "inform":
                self.extract_preferences(user_utterance)
                if self.preferences["food_type"]:
                    print(f"System: Great, you're looking for {self.preferences['food_type']} food. What's your price range?")
                    self.redirection("food_type")
                else:
                    print("System: Could you please tell me the type of food you prefer?")
            elif dialog_act in ["bye", "negate"]:
                self.state = "goodbye"
                print("System: Goodbye!")

        elif self.state == "ask_price_range":
            if dialog_act == "inform":
                self.extract_preferences(user_utterance)
                if self.preferences["price_range"]:
                    print(f"System: You're looking for a restaurant with {self.preferences['price_range']} price. I'll find the best options for you!")
                    self.redirection("price_range")
                else:
                    print("System: Could you please tell me your price range?")
            elif dialog_act in ["bye", "negate"]:
                self.state = "goodbye"
                print("System: Goodbye!")

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
                print(f"Do you want to change the {self.preferences_name[self.changes_counter]} of the restaurant? (Yes/No)")
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
