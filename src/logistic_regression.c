/**
 * @file logistic_regression.c
 * @brief Implementation of logistic regression
 */

#include "logistic_regression.h"
#include <stdlib.h>
#include <math.h>

double sigmoid(double x) {
    /* Clip to prevent overflow */
    if (x > 500.0) return 1.0;
    if (x < -500.0) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

Matrix* logreg_fit(const Matrix *X, const Matrix *y, double lr, int epochs) {
    if (!X || !y || X->rows != y->rows || lr <= 0 || epochs <= 0) {
        return NULL;
    }

    size_t n = X->rows;
    size_t m = X->cols;

    /* Initialize weights to zero */
    Matrix *weights = matrix_alloc(m, 1);
    if (!weights) {
        return NULL;
    }

    Matrix *Xt = matrix_transpose(X);
    if (!Xt) {
        matrix_free(weights);
        return NULL;
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        /* Compute linear combination: z = X * w */
        Matrix *z = matrix_matmul(X, weights);
        if (!z) {
            matrix_free(weights);
            matrix_free(Xt);
            return NULL;
        }

        /* Apply sigmoid and compute error */
        Matrix *pred = matrix_alloc(n, 1);
        if (!pred) {
            matrix_free(z);
            matrix_free(weights);
            matrix_free(Xt);
            return NULL;
        }

        for (size_t i = 0; i < n; i++) {
            pred->data[i] = sigmoid(z->data[i]);
        }
        matrix_free(z);

        /* Compute error: pred - y */
        Matrix *error = matrix_sub(pred, y);
        matrix_free(pred);
        if (!error) {
            matrix_free(weights);
            matrix_free(Xt);
            return NULL;
        }

        /* Compute gradient: X' * error / n */
        Matrix *gradient = matrix_matmul(Xt, error);
        matrix_free(error);
        if (!gradient) {
            matrix_free(weights);
            matrix_free(Xt);
            return NULL;
        }

        /* Update weights: w = w - lr * gradient / n */
        for (size_t i = 0; i < m; i++) {
            weights->data[i] -= lr * gradient->data[i] / (double)n;
        }

        matrix_free(gradient);
    }

    matrix_free(Xt);
    return weights;
}

Matrix* logreg_predict_proba(const Matrix *X, const Matrix *weights) {
    if (!X || !weights || X->cols != weights->rows) {
        return NULL;
    }

    Matrix *z = matrix_matmul(X, weights);
    if (!z) {
        return NULL;
    }

    Matrix *proba = matrix_alloc(X->rows, 1);
    if (!proba) {
        matrix_free(z);
        return NULL;
    }

    for (size_t i = 0; i < X->rows; i++) {
        proba->data[i] = sigmoid(z->data[i]);
    }

    matrix_free(z);
    return proba;
}

Matrix* logreg_predict(const Matrix *X, const Matrix *weights, double threshold) {
    Matrix *proba = logreg_predict_proba(X, weights);
    if (!proba) {
        return NULL;
    }

    Matrix *labels = matrix_alloc(X->rows, 1);
    if (!labels) {
        matrix_free(proba);
        return NULL;
    }

    for (size_t i = 0; i < X->rows; i++) {
        labels->data[i] = proba->data[i] >= threshold ? 1.0 : 0.0;
    }

    matrix_free(proba);
    return labels;
}
