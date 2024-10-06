""" Add random characteristics to the restaurant_info.csv file """
import numpy as np
import pandas as pd

restaurants_df = pd.read_csv('../data/restaurant_info.csv')

crowdedness = ['busy', 'calm']
restaurants_df['crowdedness'] = [crowdedness[i] for i in np.random.randint(0, 2, len(restaurants_df))]

length_of_stay = ['short', 'long']
restaurants_df['length_of_stay'] = [length_of_stay[i] for i in np.random.randint(0, 2, len(restaurants_df))]

assigned_seats = [True, False]
restaurants_df['assigned_seats'] = [assigned_seats[i] for i in np.random.randint(0, 2, len(restaurants_df))]

restaurants_df.to_csv('data/restaurant_info.csv', index=False)
