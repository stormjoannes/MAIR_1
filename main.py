from algorithm import Algorithm
from dialog_system import DialogManager
from restaurant_selector import RestaurantSelector

basic_dict = {
    'food_type': ['asian', 'korean', 'japanese', 'indian', 'thai'],
    'price_range': ['moderate', 'expensive', 'cheap'],
    'location': ['north', 'south', 'east', 'west', 'center']
}
dynamic_dict = {
    'food_type': set(),
    'price_range': set(),
    'location': set()
}

algorithm = Algorithm(basic_dict, dynamic_dict)
restaurant_selector = RestaurantSelector('data/restaurant_info.csv')
dialog_manager = DialogManager(algorithm, restaurant_selector)
dialog_manager.run()
