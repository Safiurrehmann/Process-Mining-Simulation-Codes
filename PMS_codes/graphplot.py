import numpy as np
import matplotlib.pyplot as plt
import math

# Function to compute the fare
def fare(x):
    base_fare = 2.50
    cost_per_quarter_mile = 0.60
    return base_fare + cost_per_quarter_mile * math.ceil(x / 0.25)

# Generate x values from 0 to 2 miles (in small increments for smooth plotting)
x_values = np.linspace(0, 2, 1000)
y_values = [fare(x) for x in x_values]

# Plotting the graph
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label="Taxi Fare ($)", color='blue')

# Adding labels and title
plt.title("Taxi Fare vs Distance Traveled", fontsize=14)
plt.xlabel("Distance Traveled (miles)", fontsize=12)
plt.ylabel("Fare ($)", fontsize=12)

# Display the plot
plt.grid(True)
plt.legend()
plt.show()
