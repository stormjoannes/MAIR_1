import pandas as pd

class RestaurantSelector:
    def __init__(self, csv_path='data/restaurant_info.csv'):
        self.restaurants_df = pd.read_csv(csv_path)

    def filter_restaurants(self, food_type=None, price_range=None, area=None):
        filtered_df = self.restaurants_df

        if food_type and food_type.lower() != 'blank':
            filtered_df = filtered_df[filtered_df['food'].str.contains(food_type, case=False, na=False)]

        if price_range and price_range.lower() != 'blank':
            filtered_df = filtered_df[filtered_df['pricerange'].str.contains(price_range, case=False, na=False)]

        if area and area.lower() != 'blank':
            filtered_df = filtered_df[filtered_df['area'].str.contains(area, case=False, na=False)]

        return filtered_df

    def apply_inference_rules(self, restaurant, user_preferences):
        # Dynamically determine which rules to apply based on user preferences
        properties = {}
        # Rule 1: Cheap and good food attracts tourists
        if user_preferences.get('price_range') == 'cheap' and restaurant['length_of_stay'] == 'short':
            properties['touristic'] = True
        # Rule 2: Romanian cuisine is not typically touristic
        if restaurant['food'] == 'Romanian':
            properties['touristic'] = False
        # Rule 3: Busy places have assigned seats
        if restaurant['crowdedness'] == 'busy':
            properties['assigned_seats'] = True
        # Rule 4: Not suitable for children if stay is long
        if restaurant['length_of_stay'] == 'long':
            properties['suitable_for_children'] = False
        # Rule 5 & 6: Romantic settings
        if restaurant['crowdedness'] == 'busy':
            properties['romantic'] = False
        if restaurant['length_of_stay'] == 'long':
            properties['romantic'] = True

        # Update restaurant info based on properties deduced from rules
        for key, value in properties.items():
            restaurant[key] = value

        return restaurant

    def recommend_restaurant(self, food_type=None, price_range=None, area=None, user_preferences=None):
        # Initial filtering based on directly available columns
        filtered_restaurants = self.filter_restaurants(food_type, price_range, area)

        # Apply inference rules to each restaurant based on user preferences
        filtered_restaurants = filtered_restaurants.apply(lambda x: self.apply_inference_rules(x, user_preferences),
                                                          axis=1)

        # Further filtering based on specific user preferences for properties like 'romantic' and 'children'
        if user_preferences:
            if user_preferences['romantic']:
                filtered_restaurants = filtered_restaurants[
                    filtered_restaurants['romantic'] == user_preferences['romantic']]
            if user_preferences['children']:
                filtered_restaurants = filtered_restaurants[
                    filtered_restaurants['children'] == user_preferences['children']]
            if user_preferences['touristic']:
                filtered_restaurants = filtered_restaurants[
                    filtered_restaurants['touristic'] == user_preferences['touristic']]
            if user_preferences['assigned_seats']:
                filtered_restaurants = filtered_restaurants[
                    filtered_restaurants['assigned_seats'] == user_preferences['assigned_seats']]

        if filtered_restaurants.empty:
            return "System: Sorry, no restaurant matches your preferences."

        return filtered_restaurants
