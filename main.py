import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = "Dataset/Nairobi Office Price Ex.csv"
data = pd.read_csv(file_path)

# Extract the relevant columns
x = data['SIZE'].values  # Office size
y = data['PRICE'].values  # Office price


# Define mean_squared error and gradient_descent functions
def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error between true and predicted values.

    Parameters:
    y_true (array): Actual target values
    y_pred (array): Predicted target values by the model

    Returns:
    float: Mean Squared Error
    """
    # Calculate the loss or cost
    cost = np.sum((y_true - y_pred) ** 2) / len(y_true)
    return cost


def gradient_descent(x, y, iterations=10, learning_rate=0.001):
    """
    Perform gradient descent to minimize MSE for a linear regression model.

    Parameters:
    x (array): Feature values (e.g., office sizes)
    y (array): Target values (e.g., office prices)
    iterations (int): Number of iterations (default=10 for quick demo)
    learning_rate (float): Learning rate for the updates (default=0.0001)

    Returns:
    tuple: Final values of slope (m) and intercept (c)
    """
    # Initialize the slope (m) and intercept (c) with random values
    m = np.random.randn()
    c = np.random.randn()
    n = len(y)  # Number of data points

    # Gradient Descent Iterations
    for i in range(iterations):
        # Predicted values
        y_pred = m * x + c

        # Calculate and print cost
        cost = mean_squared_error(y, y_pred)
        print(f"Epoch {i + 1}, Mean Squared Error: {cost:.4f}")

        # Calculate gradients
        dm = (-2 / n) * np.sum(x * (y - y_pred))  # Gradient with respect to m
        dc = (-2 / n) * np.sum(y - y_pred)  # Gradient with respect to c

        # Update m and c
        m -= learning_rate * dm
        c -= learning_rate * dc

    return m, c


# Train the model using gradient descent
learning_rate = 0.00005
epochs = 10
m, c = gradient_descent(x, y, iterations=epochs, learning_rate=learning_rate)

# Plotting the line of best fit after training
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, m * x + c, color='red', label='Best fit line')
plt.xlabel('Office Size (sq ft)')
plt.ylabel('Office Price')
plt.title('Office Size vs. Price with Best Fit Line after Training')
plt.legend()
plt.savefig("Office_Size_vs_Price_Best_Fit.png")
plt.show()

# Prediction for office size of 100 sq. ft
predicted_price_100 = m * 100 + c
print(f"Predicted office price for 100 sq. ft: {predicted_price_100:.2f}")
