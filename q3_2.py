import numpy as np
import matplotlib.pyplot as plt
from q3_1 import gradient_descent_ridge
from q1_1 import rmse, predict, data_matrix_bias
import pandas as pd


# Loading the data
x_test = pd.read_csv('X_test.csv').to_numpy()
x_train = pd.read_csv('X_train.csv').to_numpy()
y_test = pd.read_csv('y_test.csv').to_numpy().ravel()
y_train = pd.read_csv('y_train.csv').to_numpy().ravel()

# Appending the biais term to x matrices
x_test = data_matrix_bias(x_test)
x_train = data_matrix_bias(x_train)

# Doing the gradient descents for different learning rates 
results_const_schedule = gradient_descent_ridge(x_train, y_train, lamb=1.0, eta0=0.001, T= 100, schedule="constant", k_decay=0.001)
results_exp_schedule = gradient_descent_ridge(x_train, y_train, lamb=1.0, eta0=0.001,T= 100,schedule="exp_decay", k_decay=0.001,)
restults_cos_schedule = gradient_descent_ridge(x_train, y_train, lamb=1.0, eta0=0.001,T= 100, schedule="cosine", k_decay=0.001)

"""
# Plot the training loss curves w.r.t to iterations
plt.plot(results_const_schedule[1], label="Constant Learning Rate")
plt.plot(results_exp_schedule[1], label="Exponential Decay Learning Rate")
plt.plot(restults_cos_schedule[1], label="Cosine Annealing Learning Rate")
plt.xlabel("Iterations")
plt.ylabel("Training loss")
plt.title("Training loss evolution for different learning rate schedules")
plt.legend()
plt.show()
"""


# Make a table for the RMSE values for the different schedules in matplotlib
schedules = ["Constant", "Exponential Decay", "Cosine Annealing"]
print(x_test.shape)
print(results_const_schedule[0].shape)

rmse_values = [
    rmse(y_test, predict(x_test, results_const_schedule[0])),
    rmse(y_test, predict(x_test, results_exp_schedule[0])),
    rmse(y_test, predict(x_test, restults_cos_schedule[0]))
]
plt.table(cellText=[rmse_values], colLabels=schedules, loc='center')
plt.axis('off')
plt.title("RMSE values for different learning rate schedules")
plt.show()





