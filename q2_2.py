from q2_1 import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Import the data
x_train = pd.read_csv('X_train.csv').to_numpy()
y_train = pd.read_csv('y_train.csv').to_numpy().ravel()

lambda_list = [0.01, 0.1, 1, 10, 100]
n_folds = 5

# Find best lambda for each metric
results_RMSE = cross_validate_ridge(x_train, y_train, lambda_list, n_folds, "RMSE")
results_MAE = cross_validate_ridge(x_train, y_train, lambda_list, n_folds, "MAE")
results_MaxError = cross_validate_ridge(x_train, y_train, lambda_list, n_folds, "MaxError")

# Plot the results in a table 

# Build a DataFrame for display 
df_data = {
    "Lambda": lambda_list,
    "RMSE": results_RMSE[1] ,
    "MAE": results_MAE[1],
    "MaxError": results_MaxError[1],
}

df = pd.DataFrame(df_data)

# Add a row for best lambda
best_row = {
    "Lambda": "Best λ",
    "RMSE": results_RMSE[0],
    "MAE": results_MAE[0],
    "MaxError": results_MaxError[0],
}

df = pd.concat([df, pd.DataFrame([best_row])], ignore_index=True)

# Plot table
#TODO : Unvibe-code this
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)

plt.show()



