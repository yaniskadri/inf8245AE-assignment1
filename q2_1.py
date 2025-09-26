import numpy as np
from q1_1 import rmse, ridge_regression_optimize, data_matrix_bias, predict


# Part (a)
def cv_splitter(X, y, k):
    """
    Splits data into k folds for cross-validation.
    Returns a list of tuples: (X_train_fold, y_train_fold, X_val_fold, y_val_fold)
    """
   # Shuffle the data while preserving x-y correspondance
    n_samples = X.shape[0]

    # Handle exceptions 
    if n_samples != y.shape[0]:
        raise ValueError("Number of rows in X and length of y must match")
    if k <= 0 or k > n_samples:
        raise ValueError("k must be between 1 and the number of samples")
    
    #TODO : gérer les cas k = 1 et k = n_samples
    perm = np.random.permutation(n_samples)

    x_perm = X[perm]
    y_perm = y[perm]

    # Split the data into k folds
    folds = []
    start = 0
    fold_size = n_samples // k
    remainder = n_samples % k
  
    for i in range(k):
        size = fold_size + (remainder if i == 0 else 0)
        end = start + size

        x_val_fold = x_perm[start:end]
        y_val_fold = y_perm[start:end]
        x_train_fold = np.concatenate((x_perm[:start], x_perm[end:]), axis = 0)
        #x_train_fold = x_perm[np.r_[0:start, end:]]
        y_train_fold = np.concatenate((y_perm[:start], y_perm[end:]), axis = 0)

        folds.append((x_train_fold, y_train_fold, x_val_fold, y_val_fold))
        start

    #TODO test if its good or ask LLM 
    return folds



# Part (b)
def MAE(y, y_hat):
    if (y.size != y_hat.size):
        raise ValueError("y and y_hat must have the same length")
    
    if (y.size == 0 or y_hat.size == 0):
        raise ValueError("y and y_hat must not be empty")
    
    return np.mean(np.abs(y - y_hat))



def MaxError(y, y_hat):
    if (y.size != y_hat.size):
        raise ValueError("y and y_hat must have the same length")
    
    if (y.size == 0 or y_hat.size == 0):
        raise ValueError("y and y_hat must not be empty")
    
    return np.max(np.abs(y - y_hat))




# Part (c)
def cross_validate_ridge(X, y, lambda_list, k, metric):
    """
    Performs k-fold CV over lambda_list using the given metric.
    metric: one of "MAE", "MaxError", "RMSE"
    Returns the lambda with best average score and a dictionary of mean scores.
    """
    # Input validation
    if len(lambda_list) == 0:
        raise ValueError("lambda_list must not be empty")
    
    # Mapping metric input to real functions 
    metric_map = {
        "MaxError" : MaxError, 
        "MAE": MAE, 
        "RMSE": rmse
    }
    
    folds = cv_splitter(X, y, k) # Format = [(X_train_fold, y_train_fold, X_val_fold, y_val_fold), ...]
    mean_val_scores = [] 

    # Compute error on each validation fold 
    for lam in lambda_list:
        validation_metrics = []

        for fold in folds:
            # Append biais columns 
        
            x_train =data_matrix_bias(fold[0])
            x_val = data_matrix_bias(fold[2])
    

            weights = ridge_regression_optimize(x_train, fold[1], lam)
            predictions = predict(x_val, weights)
            error = metric_map[metric](fold[3], predictions)
            validation_metrics.append(error)

        mean_val_scores.append(np.mean(validation_metrics))  

    best_index = min(range(len(mean_val_scores)), key=mean_val_scores.__getitem__) # sorts the scores w.r.t value, finds the minimum and returns the index
    return (lambda_list[best_index], mean_val_scores)





# Remove the following line if you are not using it:
if __name__ == "__main__":
    # If you want to test your functions, write your code here.
    # If you write it outside this snippet, the autograder will fail!
    splits = cv_splitter(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([1, 2, 3, 4]), 2)
    print(splits)