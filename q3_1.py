import numpy as np


# Part (a)
def ridge_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, lamb: float) -> np.ndarray:
    """
    Computes the gradient of Ridge regression loss.
    ∇L(w) = -2/n X^T (y - X w) + 2 λ w
    """
    n_samples = X.shape[0]
    
    if n_samples == 0:
        raise ValueError("Input X must be non-empty") # Avoid division by zero in the next steps 
    
    if n_samples != y.shape[0]:
        raise ValueError("Number of rows in X and length of y must match")
    
    if X.shape[1] != w.shape[0]:
        raise ValueError("Number of columns in X must match length of w")
    


    left = (-2 * (X.T @ (y - X @ w))) / n_samples
    right = 2 * lamb * w

    gradient = left + right

    return gradient


# Part (b)
def learning_rate_exp_decay(eta0: float, t: int, k_decay: float) -> float:

    # Validations
    if eta0 <= 0:
        raise ValueError("Initial learning rate must be positive")
    
    if t < 0:
        raise ValueError("Time step t must be non-negative")
    
    if k_decay < 0:
        raise ValueError("Decay rate k_decay must be non-negative")
    
    # Computation
    eta_t = eta0 * np.exp(-k_decay * t)

    return eta_t



# Part (c)
def learning_rate_cosine_annealing(eta0: float, t: int, T: int) -> float:

    # Validations
    if eta0 <= 0:
        raise ValueError("Initial learning rate must be positive")
    
    if t < 0:
        raise ValueError("Time step t must be non-negative")
    
    if T <= 0:
        raise ValueError("Total time T must be positive")
    
    if t > T:
        raise ValueError("Time step t must not exceed total time T")
    

    # Computation 
    eta_t = eta0 * (1 + np.cos(np.pi * t / T)) / 2

    return eta_t


# Part (d)
def gradient_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, lamb:float, eta: float) -> np.ndarray:
    # Assuming values are validated in gradient and learning rate calculations

    gradient = ridge_gradient(X, y, w, lamb)
    w_step = w - eta * gradient

    return w_step


# Part (e)
def gradient_descent_ridge(X, y, lamb=1.0, eta0=0.1, T=500, schedule="constant", k_decay=0.01):
    # Validation of inputs
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of rows in X and length of y must match")
    
    if X.shape[0] == 0:
        raise ValueError("Input X must be non-empty")
    
    # Initialize weights with zeros (for convenience)
    n_features = X.shape[1]
    n_samples = X.shape[0]
    w = np.zeros(n_features)



    training_losses = []
    # Gradient descent loop
    for t in range(T):
        # Compute the right learning rate
        eta_t = eta0
        if schedule == "constant":
            eta_t = eta0
        elif schedule == "exp_decay":
            eta_t = learning_rate_exp_decay(eta0, t, k_decay)
        elif schedule == "cosine":
            eta_t = learning_rate_cosine_annealing(eta0, t, T)
        else:
            raise ValueError("Invalid schedule type")
        

        # Compute the gradient step
        w = gradient_step(X, y, w, lamb, eta_t)

        # Compute the loss 
        temp = y - X @ w
        temp = (temp @ temp) + lamb * (w @ w) # might cause bugs depending on shapes
        temp = temp / (2/ n_samples)
        training_losses.append(temp)

    return (w, training_losses)








# Remove the following line if you are not using it:
if __name__ == "__main__":

    # If you want to test your functions, write your code here.
    # If you write it outside this snippet, the autograder will fail!
    print("follow me on souncloud : https://soundcloud.com/foyya")
