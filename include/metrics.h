/**
 * @file metrics.h
 * @brief Evaluation metrics for ML models
 */

#ifndef METRICS_H
#define METRICS_H

#include "matrix.h"

/**
 * @brief Confusion matrix structure
 */
typedef struct {
    int tp;  /**< True Positives */
    int tn;  /**< True Negatives */
    int fp;  /**< False Positives */
    int fn;  /**< False Negatives */
} ConfusionMatrix;

/* Regression Metrics */

/**
 * @brief Mean Squared Error
 * @param y_true True values
 * @param y_pred Predicted values
 * @return MSE value
 */
double mse(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Root Mean Squared Error
 * @param y_true True values
 * @param y_pred Predicted values
 * @return RMSE value
 */
double rmse(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Mean Absolute Error
 * @param y_true True values
 * @param y_pred Predicted values
 * @return MAE value
 */
double mae(const Matrix *y_true, const Matrix *y_pred);

/* Classification Metrics */

/**
 * @brief Classification accuracy
 * @param y_true True labels
 * @param y_pred Predicted labels
 * @return Accuracy (0 to 1)
 */
double accuracy(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Precision (for binary classification)
 * @param y_true True labels (0/1)
 * @param y_pred Predicted labels (0/1)
 * @return Precision value
 */
double precision(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Recall (for binary classification)
 * @param y_true True labels (0/1)
 * @param y_pred Predicted labels (0/1)
 * @return Recall value
 */
double recall(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief F1 Score (for binary classification)
 * @param y_true True labels (0/1)
 * @param y_pred Predicted labels (0/1)
 * @return F1 score
 */
double f1_score(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Compute confusion matrix
 * @param y_true True labels (0/1)
 * @param y_pred Predicted labels (0/1)
 * @return Confusion matrix
 */
ConfusionMatrix confusion_matrix(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Print confusion matrix
 * @param cm Confusion matrix to print
 */
void confusion_matrix_print(const ConfusionMatrix *cm);

#endif /* METRICS_H */
