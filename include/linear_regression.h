/**
 * @file linear_regression.h
 * @brief Linear regression for ML library
 */

#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "matrix.h"

/**
 * @brief Fit linear regression using closed-form solution (normal equation)
 *
 * Solves: w = (X'X)^(-1) X'y
 *
 * @param X Feature matrix (n x m), should include bias column
 * @param y Target vector (n x 1)
 * @return Weight vector (m x 1), or NULL on error
 */
Matrix* linreg_fit_closed(const Matrix *X, const Matrix *y);

/**
 * @brief Fit linear regression using gradient descent
 *
 * Update rule: w = w - lr * X' * (X*w - y) / n
 *
 * @param X Feature matrix (n x m), should include bias column
 * @param y Target vector (n x 1)
 * @param lr Learning rate
 * @param epochs Number of iterations
 * @return Weight vector (m x 1), or NULL on error
 */
Matrix* linreg_fit_gd(const Matrix *X, const Matrix *y, double lr, int epochs);

/**
 * @brief Predict target values
 * @param X Feature matrix (n x m)
 * @param weights Weight vector (m x 1)
 * @return Predicted values (n x 1), or NULL on error
 */
Matrix* linreg_predict(const Matrix *X, const Matrix *weights);

#endif /* LINEAR_REGRESSION_H */
