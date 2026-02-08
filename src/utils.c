/**
 * @file utils.c
 * @brief Implementation of utility functions
 */

#include "utils.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

static int seeded = 0;

void rand_seed(unsigned int seed) {
    srand(seed);
    seeded = 1;
}

double rand_uniform(void) {
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

double rand_uniform_range(double min, double max) {
    return min + rand_uniform() * (max - min);
}

double rand_normal(void) {
    /* Box-Muller transform */
    static int have_spare = 0;
    static double spare;

    if (have_spare) {
        have_spare = 0;
        return spare;
    }

    double u, v, s;
    do {
        u = rand_uniform() * 2.0 - 1.0;
        v = rand_uniform() * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    have_spare = 1;

    return u * s;
}

double rand_normal_params(double mean, double std) {
    return mean + std * rand_normal();
}

double mean(const double *data, size_t n) {
    if (n == 0 || !data) {
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }

    return sum / (double)n;
}

double variance(const double *data, size_t n) {
    if (n <= 1 || !data) {
        return 0.0;
    }

    double m = mean(data, n);
    double sum = 0.0;

    for (size_t i = 0; i < n; i++) {
        double diff = data[i] - m;
        sum += diff * diff;
    }

    return sum / (double)(n - 1);  /* Bessel's correction */
}

double std_dev(const double *data, size_t n) {
    return sqrt(variance(data, n));
}

void shuffle_indices(size_t *indices, size_t n) {
    if (n <= 1 || !indices) {
        return;
    }

    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(rand_uniform() * (i + 1));
        size_t temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}
