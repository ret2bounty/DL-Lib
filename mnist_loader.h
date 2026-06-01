#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include "deeplearn.h"

/*
 * image_path
 * label_path
 * out_X       - (n_samples × 784) float matrix, pixels in [0,1]
 * out_y       - (n_samples × 10)  one-hot encoded matrix
 * max_samples - 0 = load all
 */
int mnist_load(const char *image_path, const char *label_path, Matrix *out_X, Matrix *out_y, int max_samples);

#endif
