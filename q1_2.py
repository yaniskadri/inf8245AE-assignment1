import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from q1_1 import (
    data_matrix_bias,
    linear_regression_optimize,
    ridge_regression_optimize,
    weighted_ridge_regression_optimize,
    predict,
    rmse
)

# Loading the data
x_test = pd.read_csv('X_test.csv').to_numpy()
x_train = pd.read_csv('X_train.csv').to_numpy()
y_test = pd.read_csv('y_test.csv').to_numpy().ravel()
y_train = pd.read_csv('y_train.csv').to_numpy().ravel()

# Appending the biais term to x matrices 
x_test_biais = data_matrix_bias(x_test)
x_train_biais = data_matrix_bias(x_train)


# 1 - Ordinary Least Squares Regression 
ols_weigts = linear_regression_optimize(x_train_biais, y_train)
ols_predictions = predict(x_test_biais, ols_weigts)
ols_rmse = rmse(y_test, ols_predictions)

# 2 - Ridge Regression for lambda = 1.0
ridge_weights = ridge_regression_optimize(x_train_biais, y_train, 1.0)
ridge_predictions = predict(x_test_biais, ridge_weights)
ridge_rmse = rmse(y_test, ridge_predictions)

# 3 - Weighted Ridge Regression for given lambda vector
lambda_vec = np.array([0.1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3])

weighted_ridge_weights = weighted_ridge_regression_optimize(x_train_biais, y_train, lambda_vec)
weighted_ridge_predictions = predict(x_test_biais, weighted_ridge_weights)
weighted_ridge_rmse = rmse(y_test, weighted_ridge_predictions)


# Plot the resuls

# OLS 
""""
plt.scatter(y_test, ols_predictions, color='crimson', marker='x')

plt.title('Ordinary Least Squares Regression - Predictions vs Actual')
plt.xlabel('Actual y values')
plt.ylabel('Predicted y values')
plt.grid()
plt.legend()
plt.show()
"""

""""
# Ridge regression
plt.scatter(y_test, ridge_predictions, color='rebeccapurple', marker='x')

plt.title('Ridge Regression - Predictions vs Actual')
plt.xlabel('Actual y values')
plt.ylabel('Predicted y values')
plt.grid()
plt.legend()
plt.show()
"""

plt.scatter(y_test, weighted_ridge_predictions, color='springgreen', marker='x')

plt.title('Weighted Ridge Regression - Predictions vs Actual')
plt.xlabel('Actual y values')
plt.ylabel('Predicted y values')
plt.grid()
plt.legend()
plt.show()










