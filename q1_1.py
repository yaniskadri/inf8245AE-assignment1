import numpy as np


# Part (a)
def data_matrix_bias(X: np.ndarray) -> np.ndarray:
    # Cover the edge cases 
    if X.size == 0:
        raise ValueError("Input X must be non-empty")
    
    if X.ndim == 1:
        X = X.reshape(-1, 1) # Convert x to a column vector if it is only a row vector
    elif X.ndim > 2:
        raise ValueError("Input X must be 1-D or 2-D")
    
    
    rows_count = X.shape[0]
    biases = np.ones((rows_count, 1), dtype=np.float64)

    return np.hstack((biases, X))


# Part (b)
def linear_regression_optimize(X: np.ndarray, y: np.ndarray) -> np.ndarray:

    if len(X) != len(y):
        raise ValueError("Number of rows in X and length of y must match")

    # We use the pseudo-inverse to handle cases where X is non-invertible cases
    w = np.linalg.pinv(X) @ y

    return w.ravel()


# Part (c)
def ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lamb: float) -> np.ndarray:

    if len(X) != len(y):
        raise ValueError("Number of examples and target values for those examples must match")

    n_features = X.shape[1]
    lambda_I = lamb * np.eye(n_features)

    left = X.T @ X + lambda_I
    right = X.T @ y 
    w = np.linalg.pinv(left) @ right

    return w.ravel()
  

# Part (e)
def weighted_ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lambda_vec) -> np.ndarray:

    if len(X) != len(y):
        raise ValueError("Number of rows in X and length of y must match")
    
    n_features = X.shape[1]

    if lambda_vec.size != n_features:
        raise ValueError("Length of lambda_vec must match number of features without biais term")

    lambda_matrix = np.diag(lambda_vec)

    left = X.T @ X + lambda_matrix
    right = X.T @ y
    w = np.linalg.pinv(left) @ right

    return w.ravel()


# Part (f)
def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    if len(X) == 0 or len(w) == 0:
        raise ValueError("X and w must be non-empty")
    
    y_hat = np.array(X) @ np.array(w)
    return y_hat.ravel()

# Part (f)
def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:

    # Handle the exception of y and y_hat not having the same size
    if len(y) == 0 or len(y_hat) == 0:
        raise ValueError("Target values and predictions must be non-empty")
    if len(y) != len(y_hat):
        raise ValueError("Target values and predictions must have the same size")

    return np.sqrt(np.mean((np.array(y) - np.array(y_hat)) ** 2))



