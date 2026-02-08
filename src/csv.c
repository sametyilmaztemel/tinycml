/**
 * @file csv.c
 * @brief Implementation of CSV loading and saving
 */

#include "csv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE_LENGTH 4096
#define INITIAL_CAPACITY 100

static int count_columns(const char *line) {
    int count = 1;
    const char *p = line;
    while (*p) {
        if (*p == ',') {
            count++;
        }
        p++;
    }
    return count;
}

static int is_empty_line(const char *line) {
    while (*line) {
        if (!isspace((unsigned char)*line)) {
            return 0;
        }
        line++;
    }
    return 1;
}

Matrix* csv_load(const char *filename, int has_header) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "csv_load: cannot open file '%s'\n", filename);
        return NULL;
    }

    char line[MAX_LINE_LENGTH];
    size_t capacity = INITIAL_CAPACITY;
    size_t rows = 0;
    size_t cols = 0;

    /* Temporary storage for values */
    double *values = malloc(capacity * sizeof(double));
    if (!values) {
        fclose(fp);
        fprintf(stderr, "csv_load: memory allocation failed\n");
        return NULL;
    }

    /* Skip header if present */
    if (has_header) {
        if (!fgets(line, sizeof(line), fp)) {
            free(values);
            fclose(fp);
            fprintf(stderr, "csv_load: file is empty\n");
            return NULL;
        }
    }

    /* Read data lines */
    while (fgets(line, sizeof(line), fp)) {
        if (is_empty_line(line)) {
            continue;
        }

        /* Determine column count from first data line */
        if (cols == 0) {
            cols = count_columns(line);
        }

        /* Parse values from line */
        char *ptr = line;
        char *end;
        size_t col = 0;

        while (*ptr && col < cols) {
            /* Skip whitespace */
            while (*ptr && isspace((unsigned char)*ptr)) {
                ptr++;
            }

            double val = strtod(ptr, &end);

            if (ptr == end) {
                /* Parse error */
                fprintf(stderr, "csv_load: parse error at row %zu, col %zu\n", rows + 1, col + 1);
                free(values);
                fclose(fp);
                return NULL;
            }

            /* Expand storage if needed */
            if (rows * cols + col >= capacity) {
                capacity *= 2;
                double *new_values = realloc(values, capacity * sizeof(double));
                if (!new_values) {
                    free(values);
                    fclose(fp);
                    fprintf(stderr, "csv_load: memory allocation failed\n");
                    return NULL;
                }
                values = new_values;
            }

            values[rows * cols + col] = val;
            col++;

            /* Move past comma */
            ptr = end;
            while (*ptr && (isspace((unsigned char)*ptr) || *ptr == ',')) {
                if (*ptr == ',') {
                    ptr++;
                    break;
                }
                ptr++;
            }
        }

        rows++;
    }

    fclose(fp);

    if (rows == 0 || cols == 0) {
        free(values);
        fprintf(stderr, "csv_load: no data found\n");
        return NULL;
    }

    /* Create matrix and copy data */
    Matrix *m = matrix_alloc(rows, cols);
    if (!m) {
        free(values);
        return NULL;
    }

    memcpy(m->data, values, rows * cols * sizeof(double));
    free(values);

    return m;
}

int csv_save(const Matrix *m, const char *filename) {
    if (!m || !filename) {
        return -1;
    }

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "csv_save: cannot open file '%s' for writing\n", filename);
        return -1;
    }

    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            fprintf(fp, "%.6f", m->data[i * m->cols + j]);
            if (j < m->cols - 1) {
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return 0;
}
