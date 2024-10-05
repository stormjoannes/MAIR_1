from dialog_system import DialogManager


def main():
    amount_of_recommendations = 3

    dialog_manager = DialogManager(amount_of_recommendations)
    dialog_manager.run()


if __name__ == "__main__":
    main()
