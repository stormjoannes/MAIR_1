import pandas as pd

class RestaurantSelector:
    def __init__(self, csv_path='data/restaurant_info.csv'):
        self.restaurants_df = pd.read_csv(csv_path)

    def filter_restaurants(self, food_type=None, price_range=None, area=None):
        filtered_df = self.restaurants_df

        if food_type:
            filtered_df = filtered_df[filtered_df['food'].str.contains(food_type, case=False, na=False)]

        if price_range:
            filtered_df = filtered_df[filtered_df['pricerange'].str.contains(price_range, case=False, na=False)]

        if area:
            filtered_df = filtered_df[filtered_df['area'].str.contains(area, case=False, na=False)]

        return filtered_df

    def apply_inference_rules(self, restaurant):
        """
        Applies inference rules to determine additional properties for a restaurant.

        Parameters:
            restaurant (pd.Series): A pandas Series containing the properties of a restaurant.

        Returns:
            pd.Series: Updated restaurant properties with inferred values.
        """
        if restaurant['cheap'] and restaurant['good_food']:
            restaurant['touristic'] = True

        if restaurant['cuisine'] == 'romanian':
            restaurant['touristic'] = False

        if restaurant['busy']:
            restaurant['assigned_seats'] = True

        if restaurant['length_of_stay'] == 'long':
            restaurant['children'] = False

        if restaurant['busy']:
            restaurant['romantic'] = False

        if restaurant['length_of_stay'] == 'long':
            restaurant['romantic'] = True

        return restaurant

    def recommend_restaurant(self, food_type=None, price_range=None, area=None):
        filtered_restaurants = self.filter_restaurants(food_type, price_range, area)

        if filtered_restaurants.empty:
            return "Sorry, no restaurant matches your preferences.", None

        # Apply inference rules to each restaurant and then select a random one for recommendation
        filtered_restaurants = filtered_restaurants.apply(self.apply_inference_rules, axis=1)

        recommendation = filtered_restaurants.sample(n=1).iloc[0]
        remaining_restaurants = filtered_restaurants.drop(recommendation.name)

        return recommendation, remaining_restaurants
