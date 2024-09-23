# State transition function for the dialog system
class DialogManager:
    def __init__(self):
        self.state = "welcome"  # Initial state
        self.preferences = {
            "area": None,
            "food_type": None,
            "price_range": None
        }

    def classify_dialog_act(self, user_utterance):
      
      input_bow = vectorizer.transform([user_utterance])
      
      # WE CAN CHANGE THE MODEL HERE to be able to determine the dialogue act
      prediction = clf_tree.predict(input_bow)
      print(prediction)
      return prediction

    def extract_preferences(self, user_utterance):
        """
        An algorithm identifying user preference statements in the sentences using pattern matching on 
        variable keywords and value keywords on utterances classified as inform, using Levenshtein edit distance if necessary.
        ------------------------------(TASK 3)----------------------------------
        """
        # Simulate preferences 
        if "east" in user_utterance:
            self.preferences["area"] = "east"
        if "italian" in user_utterance:
            self.preferences["food_type"] = "italian"
        if "cheap" in user_utterance:
            self.preferences["price_range"] = "cheap"

    def next_state(self, user_utterance):
      # Classify with the model the type of dialogue act
        dialog_act = self.classify_dialog_act(user_utterance)

        # State transition
        if self.state == "welcome":
            if dialog_act == "hello":
                print("System: Welcome! Please, tell me the area where you want to find a restaurant.")
                # Change state
                self.state = "ask_preferences"
            elif dialog_act == "inform" or "ack" or "affirm":
                self.extract_preferences(user_utterance)
                # Change state
                self.state = "ask_area" if not self.preferences["area"] else "ask_food_type"
            elif  dialog_act == "bye":
              self.state = "goodbye"
            else:
                print("System: Sorry, I didn't understand. Could you tell me your preferences again?")

        elif self.state == "ask_preferences":
            if dialog_act == "inform":
                self.extract_preferences(user_utterance)
                # Change state
                self.state = "ask_area" if not self.preferences["area"] else "ask_food_type"

        elif self.state == "ask_area":
            if dialog_act == "inform":
                self.extract_preferences(user_utterance)
                if self.preferences["area"]:
                    print(f"System: Got it, you're looking for a restaurant in {self.preferences['area']}. What kind of food do you want to eat?")
                    # Change state
                    self.state = "ask_food_type"
                else:
                    print("System: Could you please tell me the area you want to find a restaurant?")

        elif self.state == "ask_food_type":
            if dialog_act == "inform":
                self.extract_preferences(user_utterance)
                if self.preferences["food_type"]:
                    print(f"System: Great, you're looking for {self.preferences['food_type']} food. What's your price range?")
                    # Change state
                    self.state = "ask_price_range"
                else:
                    print("System: Could you please tell me the type of food you would like?")

        elif self.state == "ask_price_range":
            if dialog_act == "inform":
                self.extract_preferences(user_utterance)
                if self.preferences["price_range"]:
                    print(f"System: Alright, {self.preferences['price_range']} price range. Let me suggest a restaurant.")
                    # Change state
                    self.state = "suggest_restaurant"
                else:
                    print("System: Could you please specify the price range you're looking for?")

        elif self.state == "suggest_restaurant":
          # -------------------- TASK 4: Function to connect with CSV ----------------------------------
          # with model, extract dialogue andt + alg to get the meaning

          # Simulation
            print("System: I suggest trying 'La Trattoria' restaurant. Would you like more details such as phone number or address?")
            self.state = "request_further_details"

        elif self.state == "request_further_details":
            if dialog_act == "negate":
                print("System: Okay, have a great day! Goodbye.")
                self.state = "goodbye"
            else:
                print("System: The phone number is 123-456-7890. Do you want the address?")
                self.state = "provide_address"

        elif self.state == "provide_address":
            if dialog_act == "negate":
                print("System: Alright. Thank you, goodbye!")
                self.state = "goodbye"
            else:
                print("System: The address is 123 Elm Street. Would you like the postal code?")
                self.state = "provide_postalcode"

        elif self.state == "provide_postalcode":
            if dialog_act == "negate":
                print("System: Okay, have a nice day!")
                self.state = "goodbye"
            else:
                print("System: The postal code is ABC 123. Thank you for using the system!")
                self.state = "goodbye"

        elif self.state == "goodbye":
            print("System: Goodbye!")

    def run(self):
        # Start
        print("System: Hello! How can I help you today?")
        while self.state != "goodbye":
            user_input = input("You: ").lower()
            self.next_state(user_input)
