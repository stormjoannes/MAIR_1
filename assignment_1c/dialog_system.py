from assignment_1a.DTC import vectorizer, clf_tree
from algorithm import TextProcessor
from restaurant_selector import RestaurantSelector

class DialogManager:
    def __init__(self):
        self.state = "welcome"
        self.formality = "formal"
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

    def extract_preferences(self, user_utterance):
        if "please" in user_utterance or "could you" in user_utterance:
            self.formality = "formal"
        elif "hey" in user_utterance or "yo" in user_utterance:
            self.formality = "informal"

        categories = self.text_processor.categorize_words(user_utterance)
        for word, category in categories:
            if category == 'food_type' and not self.preferences["food_type"]:
                self.preferences["food_type"] = word
            elif category == 'price_range' and not self.preferences["price_range"]:
                self.preferences["price_range"] = word
            elif category == 'location' and not self.preferences["area"]:
                self.preferences["area"] = word

    def make_recommendation(self):
        recommendation, remaining_options = self.restaurant_selector.recommend_restaurant(
            self.preferences['food_type'],
            self.preferences['price_range'],
            self.preferences['area']
        )
        if isinstance(recommendation, str):
            print(recommendation)
            self.state = "welcome"
            self.response = False
            return

        self.restaurant = recommendation
        self.other_options = remaining_options
        print(f"System: Recommending a {self.preferences['price_range']} {self.preferences['food_type']} restaurant in {self.preferences['area']}: {recommendation['restaurantname']}. Do you want the phone number?")
        self.state = "request_further_details"

    def handle_state(self, dialog_act, user_utterance):
        if self.state == "welcome":
            print(self.get_response("welcome"))
            self.state = "ask_area"
        elif self.state == "start_over":
            self.reset_dialog()
        elif self.state in ["ask_area", "ask_food_type", "ask_price_range"]:
            self.extract_preferences(user_utterance)
            self.response = False
        elif self.state == "make_recommendation":
            self.make_recommendation()
        elif self.state == "request_further_details":
            if dialog_act == "negate":
                print("System: Okay, have a great day! Goodbye.")
                self.state = "goodbye"
            else:
                print(f"System: The phone number is {self.restaurant['phone']}. Do you want the address?")
                self.state = "provide_address"
        elif self.state in ["provide_address", "provide_postalcode"]:
            print(f"System: The postal code is {self.restaurant['postcode']}. Thank you for using the system!")
            self.state = "goodbye"
        else:
            print("System: Sorry, I didn't understand. Could you tell me your preferences again?")

    def next_state(self, user_utterance):
        dialog_act = self.classify_dialog_act(user_utterance)
        if user_utterance.lower() == "restart":
            dialog_act = "restart"
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
