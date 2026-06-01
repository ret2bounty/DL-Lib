#ifndef DEEPLEARN_H
#define DEEPLEARN_H

#include <stddef.h>

typedef enum {
    ACT_PRELU   = 0,
    ACT_SIGMOID = 1,
    ACT_TANH    = 2,
    ACT_SOFTMAX = 3
} Activation;

typedef struct {
    float   *data;
    size_t  rows;
    size_t  cols;
} Matrix;

Matrix mat_alloc(size_t rows, size_t cols);
void mat_free(Matrix *m);
void mat_zero(Matrix *m);
void mat_copy(Matrix *dst, const Matrix *src);
void mat_mul(Matrix *dst, const Matrix *a, const Matrix *b);
void mat_mul_atb(Matrix *dst, const Matrix *a, const Matrix *b);
void mat_mul_abt(Matrix *dst, const Matrix *a, const Matrix *b);

typedef struct {
    Matrix X_train, X_test;
    Matrix y_train, y_test;
} Dataset;

Dataset dl_split_dataset(const Matrix *X, const Matrix *y_onehot, float train_frac, unsigned int seed);
void dl_free_dataset(Dataset *d);

typedef struct {
    int        n_layers;       // total layers including output
    int        *layer_sizes;   // neurons per layer (array of length n_layers)
    Activation *activations;   // activation per hidden layer (array of length n_layers)
    float      prelu_alpha;    // PReLU param
    float      learning_rate;
    int        epochs;
    int        batch_size;     // mini-batch size for SGD
    unsigned   seed;
} NetConfig;

typedef struct {
    Matrix      W, b;       // params
    Matrix      dW, db;     // gradients
    Matrix      Z, A;       // forward cache
    Activation  act;
    float       prelu_alpha;
} Layer;

typedef struct {
    int     n_layers;
    Layer   *layers;
    float   lr;
} Network;

Network *dl_create_network(const NetConfig *cfg, int input_size);
void dl_free_network(Network *net);

void dl_train(Network *net, const NetConfig *cfg, const Matrix *X, const Matrix *y_onehot);
void  dl_predict(Network *net, const Matrix *X, Matrix *out_probs);
float dl_evaluate(Network *net, const Matrix *X, const Matrix *y_onehot);
float dl_cross_entropy_loss(const Matrix *probs, const Matrix *y_onehot);

#endif
