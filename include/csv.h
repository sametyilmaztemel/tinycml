/**
 * @file csv.h
 * @brief CSV file loading and saving for ML library
 */

#ifndef CSV_H
#define CSV_H

#include "matrix.h"

/**
 * @brief Load a CSV file into a Matrix
 * @param filename Path to CSV file
 * @param has_header 1 if file has header row to skip, 0 otherwise
 * @return Matrix with loaded data, or NULL on error
 */
Matrix* csv_load(const char *filename, int has_header);

/**
 * @brief Save a Matrix to a CSV file
 * @param m Matrix to save
 * @param filename Output file path
 * @return 0 on success, -1 on error
 */
int csv_save(const Matrix *m, const char *filename);

#endif /* CSV_H */
