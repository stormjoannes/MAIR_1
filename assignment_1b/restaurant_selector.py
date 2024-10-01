# restaurant_selector.py

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

    def recommend_restaurant(self, food_type=None, price_range=None, area=None):
        filtered_restaurants = self.filter_restaurants(food_type, price_range, area)

        if filtered_restaurants.empty:
            return "System: Sorry, no restaurant matches your preferences.", None


        recommendation = filtered_restaurants.sample(n=1).iloc[0]
        remaining_restaurants = filtered_restaurants.drop(recommendation.name)

        return recommendation, remaining_restaurants
