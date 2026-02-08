/**
 * @file logistic_regression.h
 * @brief Logistic regression for binary classification
 */

#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "matrix.h"

/**
 * @brief Sigmoid function
 * @param x Input value
 * @return 1 / (1 + exp(-x))
 */
double sigmoid(double x);

/**
 * @brief Fit logistic regression using gradient descent
 * @param X Feature matrix (n x m), should include bias column
 * @param y Binary target vector (n x 1), values 0 or 1
 * @param lr Learning rate
 * @param epochs Number of iterations
 * @return Weight vector (m x 1), or NULL on error
 */
Matrix* logreg_fit(const Matrix *X, const Matrix *y, double lr, int epochs);

/**
 * @brief Predict probabilities
 * @param X Feature matrix (n x m)
 * @param weights Weight vector (m x 1)
 * @return Probability vector (n x 1), or NULL on error
 */
Matrix* logreg_predict_proba(const Matrix *X, const Matrix *weights);

/**
 * @brief Predict class labels
 * @param X Feature matrix (n x m)
 * @param weights Weight vector (m x 1)
 * @param threshold Classification threshold (usually 0.5)
 * @return Class labels (n x 1), values 0 or 1, or NULL on error
 */
Matrix* logreg_predict(const Matrix *X, const Matrix *weights, double threshold);

#endif /* LOGISTIC_REGRESSION_H */
