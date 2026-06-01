#include "mnist_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static uint32_t read_be32(FILE *f)
{
    uint8_t b[4];
    fread(b, 1, 4, f);
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8)  |  (uint32_t)b[3];
}

int mnist_load(const char *image_path, const char *label_path, Matrix *out_X, Matrix *out_y, int max_samples)
{
    FILE *fi = fopen(image_path, "rb");
    FILE *fl = fopen(label_path, "rb");
    if (!fi || !fl) {
        fprintf(stderr, "Cannot open MNIST files:\n  %s\n  %s\n", image_path, label_path);
        return 1;
    }

    // Image file header
    uint32_t magic_img = read_be32(fi);
    uint32_t n_img = read_be32(fi);
    uint32_t rows = read_be32(fi);
    uint32_t cols = read_be32(fi);
    if (magic_img != 0x00000803) {
        fprintf(stderr, "Bad image magic: 0x%08X\n", magic_img);
        return 11;
    }

    // Label file header
    uint32_t magic_lbl = read_be32(fl);
    uint32_t n_lbl = read_be32(fl);
    if (magic_lbl != 0x00000801) {
        fprintf(stderr, "Bad label magic: 0x%08X\n", magic_lbl);
        return 1;
    }

    uint32_t n = n_img < n_lbl ? n_img : n_lbl;
    if (max_samples > 0 && (uint32_t)max_samples < n)
        n = (uint32_t)max_samples;

    uint32_t pixels = rows * cols;  // 28*28 = 784

    *out_X = mat_alloc(n, pixels);
    *out_y = mat_alloc(n, 10);

    uint8_t *buf = (uint8_t *)malloc(pixels);
    for (uint32_t i = 0; i < n; i++) {
        fread(buf, 1, pixels, fi);
        for (uint32_t p = 0; p < pixels; p++)
            out_X->data[i * pixels + p] = buf[p] / 255.0f;

        uint8_t label;
        fread(&label, 1, 1, fl);
        out_y->data[i * 10 + label] = 1.0f;
    }
    free(buf);
    fclose(fi);
    fclose(fl);
    printf("Loaded %u MNIST samples (%u×%u pixels)\n", n, rows, cols);
    return (int)n;
}
