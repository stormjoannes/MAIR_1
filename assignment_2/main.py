from dialog_system import DialogManager


def main():
    anthropomorphic = True
    amount_of_recommendations = 3

    if anthropomorphic:
        response_delay = 2
        language_style = "conversational"
        transparent = True
        memory = True
    else:
        response_delay = 0
        language_style = "efficient"
        transparent = False
        memory = False

    dialog_manager = DialogManager(amount_of_recommendations, response_delay, language_style, transparent, memory)
    dialog_manager.run()


if __name__ == "__main__":
    main()
