# Plot histogram for each of the columns in the historical_data.csv file
# Columns x, y, theta, v_x, v_y, omega, dist_to_food

import pandas as pd
import matplotlib.pyplot as plt

# Load the historical data
data = pd.read_csv("historical_data.csv")

# # Plot histograms for each of the columns
data.hist(bins=50, figsize=(20, 15))
plt.show()

# Print statistics for each of the columns
print(data.describe())