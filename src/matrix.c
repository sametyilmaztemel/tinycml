/**
 * @file matrix.c
 * @brief Implementation of matrix operations
 */

#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

Matrix* matrix_alloc(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return NULL;
    }

    Matrix *m = malloc(sizeof(Matrix));
    if (!m) {
        return NULL;
    }

    m->rows = rows;
    m->cols = cols;
    m->data = calloc(rows * cols, sizeof(double));

    if (!m->data) {
        free(m);
        return NULL;
    }

    return m;
}

void matrix_free(Matrix *m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

Matrix* matrix_copy(const Matrix *m) {
    if (!m) {
        return NULL;
    }

    Matrix *copy = matrix_alloc(m->rows, m->cols);
    if (!copy) {
        return NULL;
    }

    memcpy(copy->data, m->data, m->rows * m->cols * sizeof(double));
    return copy;
}

double matrix_get(const Matrix *m, size_t i, size_t j) {
    assert(m != NULL);
    assert(i < m->rows && j < m->cols);
    return m->data[i * m->cols + j];
}

void matrix_set(Matrix *m, size_t i, size_t j, double val) {
    assert(m != NULL);
    assert(i < m->rows && j < m->cols);
    m->data[i * m->cols + j] = val;
}

Matrix* matrix_add(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        return NULL;
    }

    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "matrix_add: dimension mismatch (%zu x %zu) vs (%zu x %zu)\n",
                a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Matrix *result = matrix_alloc(a->rows, a->cols);
    if (!result) {
        return NULL;
    }

    size_t n = a->rows * a->cols;
    for (size_t i = 0; i < n; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    return result;
}

Matrix* matrix_sub(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        return NULL;
    }

    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "matrix_sub: dimension mismatch (%zu x %zu) vs (%zu x %zu)\n",
                a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Matrix *result = matrix_alloc(a->rows, a->cols);
    if (!result) {
        return NULL;
    }

    size_t n = a->rows * a->cols;
    for (size_t i = 0; i < n; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }

    return result;
}

Matrix* matrix_mul(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        return NULL;
    }

    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "matrix_mul: dimension mismatch (%zu x %zu) vs (%zu x %zu)\n",
                a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Matrix *result = matrix_alloc(a->rows, a->cols);
    if (!result) {
        return NULL;
    }

    size_t n = a->rows * a->cols;
    for (size_t i = 0; i < n; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }

    return result;
}

Matrix* matrix_scale(const Matrix *m, double scalar) {
    if (!m) {
        return NULL;
    }

    Matrix *result = matrix_alloc(m->rows, m->cols);
    if (!result) {
        return NULL;
    }

    size_t n = m->rows * m->cols;
    for (size_t i = 0; i < n; i++) {
        result->data[i] = m->data[i] * scalar;
    }

    return result;
}

Matrix* matrix_matmul(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        return NULL;
    }

    if (a->cols != b->rows) {
        fprintf(stderr, "matrix_matmul: dimension mismatch (%zu x %zu) * (%zu x %zu)\n",
                a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Matrix *result = matrix_alloc(a->rows, b->cols);
    if (!result) {
        return NULL;
    }

    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            result->data[i * result->cols + j] = sum;
        }
    }

    return result;
}

Matrix* matrix_transpose(const Matrix *m) {
    if (!m) {
        return NULL;
    }

    Matrix *result = matrix_alloc(m->cols, m->rows);
    if (!result) {
        return NULL;
    }

    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            result->data[j * result->cols + i] = m->data[i * m->cols + j];
        }
    }

    return result;
}

void matrix_print(const Matrix *m) {
    if (!m) {
        printf("(null matrix)\n");
        return;
    }

    printf("Matrix (%zu x %zu):\n", m->rows, m->cols);
    for (size_t i = 0; i < m->rows; i++) {
        printf("  [");
        for (size_t j = 0; j < m->cols; j++) {
            printf("%8.4f", m->data[i * m->cols + j]);
            if (j < m->cols - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    }
}

void matrix_fill(Matrix *m, double val) {
    if (!m) {
        return;
    }

    size_t n = m->rows * m->cols;
    for (size_t i = 0; i < n; i++) {
        m->data[i] = val;
    }
}

Matrix* matrix_identity(size_t n) {
    if (n == 0) {
        return NULL;
    }

    Matrix *m = matrix_alloc(n, n);
    if (!m) {
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        m->data[i * n + i] = 1.0;
    }

    return m;
}
