import pandas as pd
import random

# Assuming the CSV file is stored as 'restaurants.csv', load it using pandas
restaurants_df = pd.read_csv('restaurants_info.csv')


# Updated filtering function to match the new CSV fields
def filter_restaurants(food_type=None, price_range=None, area=None):
    # Create a filtered DataFrame based on the criteria
    filtered_df = restaurants_df

    # Filter by food type if specified
    if food_type:
        filtered_df = filtered_df[filtered_df['food'].str.contains(food_type, case=False, na=False)]

    # Filter by price range if specified
    if price_range:
        filtered_df = filtered_df[filtered_df['pricerange'].str.contains(price_range, case=False, na=False)]

    # Filter by area if specified
    if area:
        filtered_df = filtered_df[filtered_df['area'].str.contains(area, case=False, na=False)]

    return filtered_df


# Function to recommend a restaurant and store remaining options
def recommend_restaurant(food_type=None, price_range=None, area=None):
    # Filter restaurants based on user preferences
    filtered_restaurants = filter_restaurants(food_type, price_range, area)

    # Check if any restaurants match the criteria
    if filtered_restaurants.empty:
        return "Sorry, no restaurant matches your preferences."

    # Randomly select one restaurant as recommendation
    recommendation = filtered_restaurants.sample(n=1).iloc[0]

    # Store the remaining restaurants after the recommendation
    remaining_restaurants = filtered_restaurants.drop(recommendation.name)

    return recommendation, remaining_restaurants


# Example usage: assuming the user wants a moderately priced British restaurant in the west area
user_food_type = "british"
user_price_range = "moderate"
user_area = "west"

# Call the recommendation system
recommended_restaurant, remaining_options = recommend_restaurant(user_food_type, user_price_range, user_area)

# Output the recommended restaurant
if isinstance(recommended_restaurant, str):
    print(recommended_restaurant)
else:
    print(f"We recommend {recommended_restaurant['restaurantname']} which serves {recommended_restaurant['food']} food "
          f"in the {recommended_restaurant['area']} area with a {recommended_restaurant['pricerange']} price range.")
    print(
        f"Phone: {recommended_restaurant['phone']}, Address: {recommended_restaurant['addr']}, Postcode: {recommended_restaurant['postcode']}")
