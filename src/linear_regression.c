/**
 * @file linear_regression.c
 * @brief Implementation of linear regression
 */

#include "linear_regression.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Simple matrix inversion for small matrices using Gauss-Jordan elimination */
static Matrix* matrix_inverse(const Matrix *m) {
    if (!m || m->rows != m->cols) {
        return NULL;
    }

    size_t n = m->rows;
    Matrix *augmented = matrix_alloc(n, 2 * n);
    if (!augmented) {
        return NULL;
    }

    /* Create augmented matrix [A | I] */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            augmented->data[i * 2 * n + j] = m->data[i * n + j];
        }
        augmented->data[i * 2 * n + n + i] = 1.0;
    }

    /* Gauss-Jordan elimination */
    for (size_t col = 0; col < n; col++) {
        /* Find pivot */
        size_t max_row = col;
        double max_val = fabs(augmented->data[col * 2 * n + col]);
        for (size_t row = col + 1; row < n; row++) {
            double val = fabs(augmented->data[row * 2 * n + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }

        /* Check for singularity */
        if (max_val < 1e-10) {
            matrix_free(augmented);
            return NULL;
        }

        /* Swap rows */
        if (max_row != col) {
            for (size_t j = 0; j < 2 * n; j++) {
                double temp = augmented->data[col * 2 * n + j];
                augmented->data[col * 2 * n + j] = augmented->data[max_row * 2 * n + j];
                augmented->data[max_row * 2 * n + j] = temp;
            }
        }

        /* Scale pivot row */
        double pivot = augmented->data[col * 2 * n + col];
        for (size_t j = 0; j < 2 * n; j++) {
            augmented->data[col * 2 * n + j] /= pivot;
        }

        /* Eliminate column */
        for (size_t row = 0; row < n; row++) {
            if (row != col) {
                double factor = augmented->data[row * 2 * n + col];
                for (size_t j = 0; j < 2 * n; j++) {
                    augmented->data[row * 2 * n + j] -= factor * augmented->data[col * 2 * n + j];
                }
            }
        }
    }

    /* Extract inverse from right half */
    Matrix *inverse = matrix_alloc(n, n);
    if (!inverse) {
        matrix_free(augmented);
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            inverse->data[i * n + j] = augmented->data[i * 2 * n + n + j];
        }
    }

    matrix_free(augmented);
    return inverse;
}

Matrix* linreg_fit_closed(const Matrix *X, const Matrix *y) {
    if (!X || !y || X->rows != y->rows) {
        return NULL;
    }

    /* Compute X' (transpose) */
    Matrix *Xt = matrix_transpose(X);
    if (!Xt) {
        return NULL;
    }

    /* Compute X'X */
    Matrix *XtX = matrix_matmul(Xt, X);
    if (!XtX) {
        matrix_free(Xt);
        return NULL;
    }

    /* Compute (X'X)^(-1) */
    Matrix *XtX_inv = matrix_inverse(XtX);
    if (!XtX_inv) {
        matrix_free(Xt);
        matrix_free(XtX);
        return NULL;
    }

    /* Compute X'y */
    Matrix *Xty = matrix_matmul(Xt, y);
    if (!Xty) {
        matrix_free(Xt);
        matrix_free(XtX);
        matrix_free(XtX_inv);
        return NULL;
    }

    /* Compute (X'X)^(-1) X'y */
    Matrix *weights = matrix_matmul(XtX_inv, Xty);

    matrix_free(Xt);
    matrix_free(XtX);
    matrix_free(XtX_inv);
    matrix_free(Xty);

    return weights;
}

Matrix* linreg_fit_gd(const Matrix *X, const Matrix *y, double lr, int epochs) {
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
        /* Compute predictions: X * w */
        Matrix *pred = matrix_matmul(X, weights);
        if (!pred) {
            matrix_free(weights);
            matrix_free(Xt);
            return NULL;
        }

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

Matrix* linreg_predict(const Matrix *X, const Matrix *weights) {
    if (!X || !weights || X->cols != weights->rows) {
        return NULL;
    }

    return matrix_matmul(X, weights);
}
