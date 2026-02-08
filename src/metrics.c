/**
 * @file metrics.c
 * @brief Implementation of evaluation metrics
 */

#include "metrics.h"
#include <stdio.h>
#include <math.h>

double mse(const Matrix *y_true, const Matrix *y_pred) {
    if (!y_true || !y_pred) {
        return 0.0;
    }

    size_t n = y_true->rows * y_true->cols;
    if (n != y_pred->rows * y_pred->cols || n == 0) {
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = y_true->data[i] - y_pred->data[i];
        sum += diff * diff;
    }

    return sum / (double)n;
}

double rmse(const Matrix *y_true, const Matrix *y_pred) {
    return sqrt(mse(y_true, y_pred));
}

double mae(const Matrix *y_true, const Matrix *y_pred) {
    if (!y_true || !y_pred) {
        return 0.0;
    }

    size_t n = y_true->rows * y_true->cols;
    if (n != y_pred->rows * y_pred->cols || n == 0) {
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += fabs(y_true->data[i] - y_pred->data[i]);
    }

    return sum / (double)n;
}

double accuracy(const Matrix *y_true, const Matrix *y_pred) {
    if (!y_true || !y_pred) {
        return 0.0;
    }

    size_t n = y_true->rows * y_true->cols;
    if (n != y_pred->rows * y_pred->cols || n == 0) {
        return 0.0;
    }

    int correct = 0;
    for (size_t i = 0; i < n; i++) {
        /* Compare as integers for classification */
        if ((int)y_true->data[i] == (int)y_pred->data[i]) {
            correct++;
        }
    }

    return (double)correct / (double)n;
}

ConfusionMatrix confusion_matrix(const Matrix *y_true, const Matrix *y_pred) {
    ConfusionMatrix cm = {0, 0, 0, 0};

    if (!y_true || !y_pred) {
        return cm;
    }

    size_t n = y_true->rows * y_true->cols;
    if (n != y_pred->rows * y_pred->cols) {
        return cm;
    }

    for (size_t i = 0; i < n; i++) {
        int actual = (int)y_true->data[i];
        int predicted = (int)y_pred->data[i];

        if (actual == 1 && predicted == 1) {
            cm.tp++;
        } else if (actual == 0 && predicted == 0) {
            cm.tn++;
        } else if (actual == 0 && predicted == 1) {
            cm.fp++;
        } else if (actual == 1 && predicted == 0) {
            cm.fn++;
        }
    }

    return cm;
}

double precision(const Matrix *y_true, const Matrix *y_pred) {
    ConfusionMatrix cm = confusion_matrix(y_true, y_pred);
    int denominator = cm.tp + cm.fp;
    if (denominator == 0) {
        return 0.0;
    }
    return (double)cm.tp / (double)denominator;
}

double recall(const Matrix *y_true, const Matrix *y_pred) {
    ConfusionMatrix cm = confusion_matrix(y_true, y_pred);
    int denominator = cm.tp + cm.fn;
    if (denominator == 0) {
        return 0.0;
    }
    return (double)cm.tp / (double)denominator;
}

double f1_score(const Matrix *y_true, const Matrix *y_pred) {
    double p = precision(y_true, y_pred);
    double r = recall(y_true, y_pred);

    if (p + r == 0.0) {
        return 0.0;
    }

    return 2.0 * (p * r) / (p + r);
}

void confusion_matrix_print(const ConfusionMatrix *cm) {
    if (!cm) {
        return;
    }

    printf("Confusion Matrix:\n");
    printf("                 Predicted\n");
    printf("              Neg      Pos\n");
    printf("Actual Neg  %5d    %5d\n", cm->tn, cm->fp);
    printf("       Pos  %5d    %5d\n", cm->fn, cm->tp);
    printf("\n");
    printf("TP: %d, TN: %d, FP: %d, FN: %d\n", cm->tp, cm->tn, cm->fp, cm->fn);
}
