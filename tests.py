import numpy as np
import pytest
import warnings
from numpy.testing import assert_array_almost_equal, assert_almost_equal

# Import all functions to test
from q1_1 import (
    data_matrix_bias,
    linear_regression_optimize,
    ridge_regression_optimize,
    weighted_ridge_regression_optimize,
    predict,
    rmse
)

from q2_1 import (
    cv_splitter,
    MAE,
    MaxError,
    cross_validate_ridge
)

from q3_1 import (
    ridge_gradient,
    learning_rate_exp_decay,
    learning_rate_cosine_annealing,
    gradient_step,
    gradient_descent_ridge
)


class TestDataMatrixBias:
    """Test suite for data_matrix_bias function"""
    
    def test_2d_matrix(self):
        """Test with standard 2D matrix"""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        result = data_matrix_bias(X)
        expected = np.array([[1, 1, 2], [1, 3, 4], [1, 5, 6]])
        assert_array_almost_equal(result, expected)
    
    def test_1d_array(self):
        """Test with 1D array (should convert to column vector)"""
        X = np.array([1, 2, 3])
        result = data_matrix_bias(X)
        expected = np.array([[1, 1], [1, 2], [1, 3]])
        assert_array_almost_equal(result, expected)
    
    def test_empty_array(self):
        """Test with empty array (should raise error)"""
        X = np.array([])
        with pytest.raises(ValueError, match="Input X must be non-empty"):
            data_matrix_bias(X)
    
    def test_3d_array(self):
        """Test with 3D array (should raise error)"""
        X = np.array([[[1, 2]], [[3, 4]]])
        with pytest.raises(ValueError, match="Input X must be 1-D or 2-D"):
            data_matrix_bias(X)
    
    def test_single_sample(self):
        """Test with single sample"""
        X = np.array([[2, 3]])
        result = data_matrix_bias(X)
        expected = np.array([[1, 2, 3]])
        assert_array_almost_equal(result, expected)


class TestLinearRegression:
    """Test suite for linear_regression_optimize function"""
    
    def test_simple_linear_fit(self):
        """Test with simple linear relationship"""
        # y = 2 + 3*x
        X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])  # With bias
        y = np.array([5, 8, 11, 14])
        w = linear_regression_optimize(X, y)
        assert_almost_equal(w[0], 2, decimal=5)  # Intercept
        assert_almost_equal(w[1], 3, decimal=5)  # Slope
    
    def test_perfect_fit(self):
        """Test with perfect linear fit"""
        X = np.array([[1, 0], [1, 1], [1, 2]])
        y = np.array([1, 2, 3])
        w = linear_regression_optimize(X, y)
        predictions = predict(X, w)
        assert_array_almost_equal(predictions, y)
    
    def test_multivariate(self):
        """Test with multiple features"""
        # y = 1 + 2*x1 + 3*x2
        X = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 2, 1]])
        y = np.array([3, 4, 6, 8])
        w = linear_regression_optimize(X, y)
        assert_almost_equal(w[0], 1, decimal=5)
        assert_almost_equal(w[1], 2, decimal=5)
        assert_almost_equal(w[2], 3, decimal=5)
    
    def test_mismatched_dimensions(self):
        """Test with mismatched X and y dimensions"""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Number of rows in X and length of y must match"):
            linear_regression_optimize(X, y)
    
    def test_underdetermined_system(self):
        """Test with more features than samples (uses pseudo-inverse)"""
        X = np.array([[1, 2, 3, 4]])
        y = np.array([5])
        w = linear_regression_optimize(X, y)
        assert w.shape == (4,)


class TestRidgeRegression:
    """Test suite for ridge_regression_optimize function"""
    
    def test_lambda_zero_equals_ols(self):
        """Ridge with lambda=0 should equal OLS"""
        X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        y = np.array([2, 4, 6, 8])
        w_ols = linear_regression_optimize(X, y)
        w_ridge = ridge_regression_optimize(X, y, 0)
        assert_array_almost_equal(w_ols, w_ridge, decimal=5)
    
    def test_regularization_effect(self):
        """Test that regularization shrinks weights"""
        X = np.array([[1, 2], [1, 3], [1, 4]])
        y = np.array([5, 7, 9])
        w_no_reg = ridge_regression_optimize(X, y, 0)
        w_reg = ridge_regression_optimize(X, y, 10)
        # Weights should be smaller with regularization
        assert np.linalg.norm(w_reg) < np.linalg.norm(w_no_reg)
    
    def test_large_lambda(self):
        """Test with very large lambda (weights should approach zero)"""
        X = np.array([[1, 1], [1, 2], [1, 3]])
        y = np.array([1, 2, 3])
        w = ridge_regression_optimize(X, y, 1e10)
        assert_array_almost_equal(w, np.zeros_like(w), decimal=3)
    
    def test_negative_lambda(self):
        """Test with negative lambda (mathematically valid but unusual)"""
        X = np.array([[1, 1], [1, 2]])
        y = np.array([2, 4])
        # Should not raise error, but result may be unstable
        w = ridge_regression_optimize(X, y, -0.1)
        assert w.shape == (2,)


class TestWeightedRidgeRegression:
    """Test suite for weighted_ridge_regression_optimize function"""
    
    def test_uniform_weights_equals_ridge(self):
        """Uniform lambda vector should equal regular ridge"""
        X = np.array([[1, 1, 2], [1, 2, 3], [1, 3, 4]])
        y = np.array([5, 8, 11])
        lamb = 2.0
        lambda_vec = np.array([lamb, lamb, lamb])
        w_ridge = ridge_regression_optimize(X, y, lamb)
        w_weighted = weighted_ridge_regression_optimize(X, y, lambda_vec)
        assert_array_almost_equal(w_ridge, w_weighted, decimal=5)
    
    def test_selective_regularization(self):
        """Test regularizing only specific features"""
        X = np.array([[1, 1, 1], [1, 2, 2], [1, 3, 3]])
        y = np.array([3, 6, 9])
        # Only regularize second feature
        lambda_vec = np.array([0, 10, 0])
        w = weighted_ridge_regression_optimize(X, y, lambda_vec)
        # Second weight should be smaller due to regularization
        assert abs(w[1]) < abs(w[2])
    
    def test_mismatched_lambda_dimensions(self):
        """Test with wrong lambda vector size"""
        X = np.array([[1, 2, 3]])
        y = np.array([4])
        lambda_vec = np.array([1, 2])  # Wrong size
        with pytest.raises(ValueError, match="Length of lambda_vec must match number of features"):
            weighted_ridge_regression_optimize(X, y, lambda_vec)


class TestPredictAndRMSE:
    """Test suite for predict and rmse functions"""
    
    def test_predict_simple(self):
        """Test simple prediction"""
        X = np.array([[1, 2], [1, 3]])
        w = np.array([1, 2])
        y_hat = predict(X, w)
        expected = np.array([5, 7])  # 1*1 + 2*2 = 5, 1*1 + 3*2 = 7
        assert_array_almost_equal(y_hat, expected)
    
    def test_predict_empty(self):
        """Test prediction with empty inputs"""
        with pytest.raises(ValueError, match="X and w must be non-empty"):
            predict(np.array([]), np.array([1, 2]))
    
    def test_rmse_zero_error(self):
        """Test RMSE with perfect predictions"""
        y = np.array([1, 2, 3])
        y_hat = np.array([1, 2, 3])
        error = rmse(y, y_hat)
        assert_almost_equal(error, 0)
    
    def test_rmse_calculation(self):
        """Test RMSE calculation"""
        y = np.array([1, 2, 3])
        y_hat = np.array([2, 2, 4])
        # Errors: 1, 0, 1 -> Mean squared: 2/3 -> RMSE: sqrt(2/3)
        expected = np.sqrt(2/3)
        assert_almost_equal(rmse(y, y_hat), expected)
    
    def test_rmse_mismatched_sizes(self):
        """Test RMSE with mismatched sizes"""
        y = np.array([1, 2])
        y_hat = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Target values and predictions must have the same size"):
            rmse(y, y_hat)


class TestCVSplitter:
    """Test suite for cv_splitter function"""
    
    def test_fold_sizes(self):
        """Test that folds are properly sized"""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([1, 2, 3, 4, 5])
        k = 3
        folds = cv_splitter(X, y, k)
        
        assert len(folds) == k
        # Check each fold
        total_val_samples = 0
        for X_train, y_train, X_val, y_val in folds:
            assert len(X_val) + len(X_train) == len(X)
            assert len(y_val) + len(y_train) == len(y)
            total_val_samples += len(X_val)
        assert total_val_samples == len(X)
    
    def test_no_data_loss(self):
        """Test that all data is preserved in splits"""
        X = np.array([[i, i+1] for i in range(10)])
        y = np.arange(10)
        k = 5
        folds = cv_splitter(X, y, k)
        
        # Collect all validation data
        all_val_data = []
        for _, _, X_val, _ in folds:
            all_val_data.extend(X_val.tolist())
        
        # Should have all original data (order may differ)
        assert len(all_val_data) == len(X)
    
    def test_invalid_k(self):
        """Test with invalid k values"""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])
        
        with pytest.raises(ValueError):
            cv_splitter(X, y, 0)  # k <= 0
        
        with pytest.raises(ValueError):
            cv_splitter(X, y, 3)  # k > n_samples


class TestErrorMetrics:
    """Test suite for MAE and MaxError functions"""
    
    def test_mae_calculation(self):
        """Test MAE calculation"""
        y = np.array([1, 2, 3, 4])
        y_hat = np.array([1.5, 2.5, 2.5, 3.5])
        # Errors: 0.5, 0.5, 0.5, 0.5 -> MAE: 0.5
        assert_almost_equal(MAE(y, y_hat), 0.5)
    
    def test_max_error_calculation(self):
        """Test MaxError calculation"""
        y = np.array([1, 2, 3, 4])
        y_hat = np.array([1.1, 2.5, 2.8, 5])
        # Errors: 0.1, 0.5, 0.2, 1 -> MaxError: 1
        assert_almost_equal(MaxError(y, y_hat), 1)
    
    def test_error_metrics_empty(self):
        """Test error metrics with empty arrays"""
        with pytest.raises(ValueError, match="y and y_hat must not be empty"):
            MAE(np.array([]), np.array([]))
        
        with pytest.raises(ValueError, match="y and y_hat must not be empty"):
            MaxError(np.array([]), np.array([]))


class TestCrossValidateRidge:
    """Test suite for cross_validate_ridge function"""
    
    def test_basic_cv(self):
        """Test basic cross-validation functionality"""
        np.random.seed(42)
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        lambda_list = [0.1, 1.0, 10.0]
        k = 5
        
        best_lambda, scores = cross_validate_ridge(X, y, lambda_list, k, "RMSE")
        
        assert best_lambda in lambda_list
        assert len(scores) == len(lambda_list)
        assert all(s >= 0 for s in scores)  # RMSE is non-negative
    
    def test_different_metrics(self):
        """Test with different metrics"""
        np.random.seed(42)
        X = np.random.randn(15, 2)
        y = np.random.randn(15)
        lambda_list = [1.0]
        k = 3
        
        for metric in ["RMSE", "MAE", "MaxError"]:
            best_lambda, scores = cross_validate_ridge(X, y, lambda_list, k, metric)
            assert len(scores) == 1
            assert scores[0] >= 0


class TestGradientDescent:
    """Test suite for gradient descent functions"""
    
    def test_ridge_gradient_calculation(self):
        """Test gradient calculation for ridge regression"""
        X = np.array([[1, 1], [1, 2]])
        y = np.array([2, 3])
        w = np.array([0.5, 0.5])
        lamb = 1.0
        
        grad = ridge_gradient(X, y, w, lamb)
        assert grad.shape == w.shape
    
    def test_ridge_gradient_at_optimum(self):
        """Gradient should be near zero at optimum"""
        X = np.array([[1, 1], [1, 2], [1, 3]])
        y = np.array([2, 4, 6])
        lamb = 0.1
        
        # Get optimal weights
        w_opt = ridge_regression_optimize(X, y, lamb)
        grad = ridge_gradient(X, y, w_opt, lamb)
        
        # Gradient should be close to zero
        assert np.linalg.norm(grad) < 1e-5
    
    def test_learning_rate_exp_decay(self):
        """Test exponential decay learning rate"""
        eta0 = 1.0
        k_decay = 0.1
        
        # Should decay exponentially
        eta_0 = learning_rate_exp_decay(eta0, 0, k_decay)
        eta_10 = learning_rate_exp_decay(eta0, 10, k_decay)
        eta_20 = learning_rate_exp_decay(eta0, 20, k_decay)
        
        assert_almost_equal(eta_0, 1.0)
        assert eta_10 < eta_0
        assert eta_20 < eta_10
        assert_almost_equal(eta_10, np.exp(-1))  # At t=10, k=0.1
    
    def test_learning_rate_cosine_annealing(self):
        """Test cosine annealing learning rate"""
        eta0 = 1.0
        T = 100
        
        # At t=0, should be eta0
        assert_almost_equal(learning_rate_cosine_annealing(eta0, 0, T), 1.0)
        
        # At t=T, should be 0
        assert_almost_equal(learning_rate_cosine_annealing(eta0, T, T), 0)
        
        # At t=T/2, should be eta0/2
        assert_almost_equal(learning_rate_cosine_annealing(eta0, T//2, T), 0.5, decimal=2)
    
    def test_gradient_step(self):
        """Test single gradient step"""
        X = np.array([[1, 1], [1, 2]])
        y = np.array([2, 4])
        w = np.array([0, 0])
        lamb = 0.1
        eta = 0.1
        
        w_new = gradient_step(X, y, w, lamb, eta)
        assert w_new.shape == w.shape
        # Should move in direction of reducing loss
        assert not np.array_equal(w_new, w)
    
    def test_gradient_descent_convergence(self):
        """Test that gradient descent converges"""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        X = data_matrix_bias(X)  # Add bias
        y = X @ np.array([1, 2, 3, 4]) + 0.1 * np.random.randn(50)
        
        # Run gradient descent
        w, losses = gradient_descent_ridge(X, y, lamb=0.1, eta0=0.01, T=500, schedule="constant")
        
        # Check convergence
        assert len(losses) == 500
        assert losses[-1] < losses[0]  # Loss should decrease
        
        # Compare with closed-form solution
        w_closed = ridge_regression_optimize(X, y, 0.1)
        assert_array_almost_equal(w, w_closed, decimal=2)
    
    def test_gradient_descent_schedules(self):
        """Test different learning rate schedules"""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        X = data_matrix_bias(X)
        y = np.random.randn(30)
        
        schedules = ["constant", "exp_decay", "cosine"]
        results = {}
        
        for schedule in schedules:
            w, losses = gradient_descent_ridge(X, y, lamb=1.0, eta0=0.01, T=100, 
                                              schedule=schedule, k_decay=0.01)
            results[schedule] = (w, losses)
            
            # All should converge
            assert losses[-1] < losses[0]
        
        # Results should be similar but not identical
        for s1 in schedules:
            for s2 in schedules:
                if s1 != s2:
                    assert not np.array_equal(results[s1][0], results[s2][0])


class TestIntegration:
    """Integration tests for functions working together"""
    
    def test_full_pipeline_ols(self):
        """Test complete OLS pipeline"""
        # Generate synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        true_w = np.array([2, -1, 3, 0.5])  # Including bias
        X_train_bias = data_matrix_bias(X_train)
        y_train = X_train_bias @ true_w + 0.1 * np.random.randn(100)
        
        X_test = np.random.randn(20, 3)
        X_test_bias = data_matrix_bias(X_test)
        y_test = X_test_bias @ true_w + 0.1 * np.random.randn(20)
        
        # Train model
        w = linear_regression_optimize(X_train_bias, y_train)
        
        # Make predictions
        y_pred = predict(X_test_bias, w)
        
        # Calculate error
        error = rmse(y_test, y_pred)
        
        # Should have reasonable error
        assert error < 1.0
        
        # Weights should be close to true weights
        assert_array_almost_equal(w, true_w, decimal=1)
    
    def test_full_pipeline_ridge_vs_ols(self):
        """Test that Ridge performs better with multicollinearity"""
        np.random.seed(42)
        n_samples = 50
        
        # Create highly correlated features
        X1 = np.random.randn(n_samples, 1)
        X2 = X1 + 0.01 * np.random.randn(n_samples, 1)  # Almost identical to X1
        X = np.hstack([X1, X2])
        X_bias = data_matrix_bias(X)
        
        # True relationship
        y = 3 * X1.ravel() + 0.5 * np.random.randn(n_samples)
        
        # OLS should be unstable
        w_ols = linear_regression_optimize(X_bias, y)
        
        # Ridge should be more stable
        w_ridge = ridge_regression_optimize(X_bias, y, lamb=1.0)
        
        # Ridge weights should be smaller in magnitude
        assert np.linalg.norm(w_ridge[1:]) < np.linalg.norm(w_ols[1:])
    
    def test_cv_selects_good_lambda(self):
        """Test that CV selects reasonable lambda"""
        np.random.seed(42)
        
        # Generate data with known good regularization
        X = np.random.randn(100, 5)
        true_w = np.array([1, 2, 0.1, 0.1, 0.1, 0.1])  # Some small weights
        X_bias = data_matrix_bias(X)
        y = X_bias @ true_w + 0.5 * np.random.randn(100)
        
        # Test range of lambdas
        lambda_list = [0.001, 0.01, 0.1, 1, 10, 100]
        
        best_lambda, scores = cross_validate_ridge(X, y, lambda_list, k=5, metric="RMSE")
        
        # Should not select extreme values
        assert best_lambda not in [0.001, 100]
    
    def test_gradient_descent_matches_closed_form(self):
        """Test that gradient descent solution matches closed-form"""
        np.random.seed(42)
        X = np.random.randn(50, 4)
        X = data_matrix_bias(X)
        y = np.random.randn(50)
        lamb = 1.0
        
        # Closed-form solution
        w_closed = ridge_regression_optimize(X, y, lamb)
        
        # Gradient descent solution
        w_gd, _ = gradient_descent_ridge(X, y, lamb=lamb, eta0=0.01, T=1000, schedule="constant")
        
        # Should be very close
        assert_array_almost_equal(w_closed, w_gd, decimal=3)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_single_sample(self):
        """Test with single training sample"""
        X = np.array([[1, 2]])
        y = np.array([3])
        
        # Should work but may overfit
        w = linear_regression_optimize(X, y)
        assert w.shape == (2,)
        
        # Perfect fit on training data
        y_pred = predict(X, w)
        assert_almost_equal(y_pred[0], 3)
    
    def test_perfect_collinearity(self):
        """Test with perfectly collinear features"""
        X = np.array([[1, 1, 2], [1, 2, 4], [1, 3, 6]])  # Third column = 2 * second
        y = np.array([1, 2, 3])
        
        # Should still work with pseudo-inverse
        w = linear_regression_optimize(X, y)
        assert not np.any(np.isnan(w))
    
    def test_very_large_values(self):
        """Test numerical stability with large values"""
        X = np.array([[1, 1e10], [1, 2e10]])
        y = np.array([1e10, 2e10])
        
        # Should handle without overflow
        w = ridge_regression_optimize(X, y, 1.0)
        assert not np.any(np.isnan(w))
        assert not np.any(np.isinf(w))
    
    def test_zero_variance_feature(self):
        """Test with constant feature"""
        X = np.array([[1, 5], [2, 5], [3, 5]])  # Second feature is constant
        X = data_matrix_bias(X)
        y = np.array([1, 2, 3])
        
        w = linear_regression_optimize(X, y)
        # Constant feature should have near-zero weight
        assert abs(w[2]) < 1e-10


class TestMathematicalProperties:
    """Test mathematical properties and invariants"""
    
    def test_ridge_reduces_to_ols(self):
        """Ridge with lambda=0 should equal OLS exactly"""
        np.random.seed(42)
        X = np.random.randn(20, 3)
        X = data_matrix_bias(X)
        y = np.random.randn(20)
        
        w_ols = linear_regression_optimize(X, y)
        w_ridge = ridge_regression_optimize(X, y, 0)
        
        assert_array_almost_equal(w_ols, w_ridge, decimal=10)
    
    def test_gradient_descent_loss_monotonic(self):
        """Loss should monotonically decrease with small learning rate"""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        X = data_matrix_bias(X)
        y = np.random.randn(30)
        
        # Very small learning rate ensures monotonic decrease
        _, losses = gradient_descent_ridge(X, y, lamb=1.0, eta0=0.001, T=100, schedule="constant")
        
        # Check monotonic decrease (allowing small numerical errors)
        for i in range(1, len(losses)):
            assert losses[i] <= losses[i-1] * (1 + 1e-6)
    
    def test_cv_variance_with_random_splits(self):
        """CV results should vary with different random seeds"""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        lambda_list = [1.0]
        
        # Run CV with different implicit random seeds
        scores1 = []
        scores2 = []
        
        for _ in range(5):
            _, s1 = cross_validate_ridge(X, y, lambda_list, k=5, metric="RMSE")
            _, s2 = cross_validate_ridge(X, y, lambda_list, k=5, metric="RMSE")
            scores1.append(s1[0])
            scores2.append(s2[0])
        
        # Scores should vary due to different splits
        assert np.var(scores1 + scores2) > 0
    
    def test_weighted_ridge_feature_importance(self):
        """Test that higher lambda reduces specific feature importance"""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X = data_matrix_bias(X)
        
        # Make y strongly dependent on second feature (index 2 after bias)
        y = 5 * X[:, 2] + 0.1 * np.random.randn(100)
        
        # High regularization on unimportant features
        lambda_vec = np.array([0, 10, 0.1, 10])  # Low reg on feature 2
        
        w = weighted_ridge_regression_optimize(X, y, lambda_vec)
        
        # Second feature should have highest weight
        assert abs(w[2]) > abs(w[1])
        assert abs(w[2]) > abs(w[3])


# Performance and stress tests
class TestPerformance:
    """Test performance with larger datasets"""
    
    def test_large_dataset_ols(self):
        """Test OLS with larger dataset"""
        np.random.seed(42)
        X = np.random.randn(1000, 10)
        X = data_matrix_bias(X)
        y = np.random.randn(1000)
        
        # Should complete without errors
        w = linear_regression_optimize(X, y)
        assert w.shape == (11,)
    
    def test_gradient_descent_convergence_speed(self):
        """Test convergence speed of different schedules"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X = data_matrix_bias(X)
        y = np.random.randn(100)
        
        schedules_convergence = {}
        
        for schedule in ["constant", "exp_decay", "cosine"]:
            _, losses = gradient_descent_ridge(X, y, lamb=1.0, eta0=0.1, T=200,
                                              schedule=schedule, k_decay=0.02)
            
            # Find iteration where loss stabilizes (change < 0.001)
            for i in range(10, len(losses)):
                if abs(losses[i] - losses[i-1]) < 0.001:
                    schedules_convergence[schedule] = i
                    break
            else:
                schedules_convergence[schedule] = len(losses)
        
        # Exp decay and cosine should converge faster than constant
        assert schedules_convergence["exp_decay"] <= schedules_convergence["constant"]
        assert schedules_convergence["cosine"] <= schedules_convergence["constant"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])