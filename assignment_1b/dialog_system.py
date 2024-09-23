class DialogManager:
    def __init__(self, algorithm, restaurant_selector):
        self.algorithm = algorithm
        self.restaurant_selector = restaurant_selector
        self.state = "welcome"
        self.preferences = {"area": None, "food_type": None, "price_range": None}
        self.restaurant = None
        self.other_options = None

    def extract_preferences(self, user_utterance):
        categories = self.algorithm.categorize_words(user_utterance)
        for word, category in categories:
            if category == 'food_type' and not self.preferences["food_type"]:
                self.preferences["food_type"] = word
            elif category == 'price_range' and not self.preferences["price_range"]:
                self.preferences["price_range"] = word
            elif category == 'location' and not self.preferences["area"]:
                self.preferences["area"] = word

    def next_state(self, user_utterance):
        if self.state == "welcome":
            self.extract_preferences(user_utterance)
            self.state = "make_recommendation"

        elif self.state == "make_recommendation":
            recommendation, remaining_options = self.restaurant_selector.recommend_restaurant(
                self.preferences['food_type'], self.preferences['price_range'], self.preferences['area']
            )
            if recommendation:
                self.restaurant = recommendation
                self.other_options = remaining_options
                print(f"Recommended: {recommendation['restaurantname']}. Want more details?")
                self.state = "provide_details"
            else:
                print("No restaurant matches your preferences.")
                self.state = "welcome"

    def run(self):
        print("System: Hello! How can I help you today?")
        while self.state != "goodbye":
            user_input = input("You: ").lower()
            self.next_state(user_input)
