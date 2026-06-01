#include "deeplearn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// MATRIX OPERATIONS

Matrix mat_alloc(size_t rows, size_t cols)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float *)calloc(rows * cols, sizeof(float));
    if (!m.data) { perror("mat_alloc"); exit(1); }
    return m;
}

void mat_free(Matrix *m)
{
    free(m->data);
    m->data = NULL;
    m->rows = m->cols = 0;
}

void mat_zero(Matrix *m)
{
    memset(m->data, 0, m->rows * m->cols * sizeof(float));
}

void mat_copy(Matrix *dst, const Matrix *src)
{
    memcpy(dst->data, src->data, src->rows * src->cols * sizeof(float));
}

void mat_mul(Matrix *dst, const Matrix *a, const Matrix *b)
{
    size_t M = a->rows, K = a->cols, N = b->cols;
    mat_zero(dst);
    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < K; k++) {
            float aik = a->data[i * K + k];
            for (size_t j = 0; j < N; j++)
                dst->data[i * N + j] += aik * b->data[k * N + j];
        }
}

void mat_mul_atb(Matrix *dst, const Matrix *a, const Matrix *b)
{
    size_t K = a->rows, M = a->cols, N = b->cols;
    mat_zero(dst);
    for (size_t i = 0; i < K; i++)
        for (size_t k = 0; k < M; k++) {
            float aik = a->data[i * M + k];
            for (size_t j = 0; j < N; j++)
                dst->data[k * N + j] += aik * b->data[i * N + j];
        }
}

void mat_mul_abt(Matrix *dst, const Matrix *a, const Matrix *b)
{
    size_t M = a->rows, N = a->cols, K = b->rows;
    mat_zero(dst);
    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < K; k++) {
            float sum = 0.f;
            for (size_t j = 0; j < N; j++)
                sum += a->data[i * N + j] * b->data[k * N + j];
            dst->data[i * K + k] = sum;
        }
}

//DATASET SPLIT

Dataset dl_split_dataset(const Matrix *X, const Matrix *y_onehot, float train_frac, unsigned int seed)
{
    size_t n = X->rows;
    size_t n_train = (size_t)(n * train_frac);
    size_t n_test = n - n_train;

    // Shuffle indices
    size_t *idx = (size_t *)malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) idx[i] = i;
    srand(seed);
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)rand() % (i + 1);
        size_t tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }

    Dataset d;
    d.X_train = mat_alloc(n_train, X->cols);
    d.X_test = mat_alloc(n_test, X->cols);
    d.y_train = mat_alloc(n_train, y_onehot->cols);
    d.y_test = mat_alloc(n_test, y_onehot->cols);

    for (size_t i = 0; i < n_train; i++) {
        memcpy(&d.X_train.data[i * X->cols], &X->data[idx[i] * X->cols], X->cols * sizeof(float));
        memcpy(&d.y_train.data[i * y_onehot->cols], &y_onehot->data[idx[i] * y_onehot->cols], y_onehot->cols * sizeof(float));
    }
    for (size_t i = 0; i < n_test; i++) {
        memcpy(&d.X_test.data[i * X->cols], &X->data[idx[n_train + i] * X->cols], X->cols * sizeof(float));
        memcpy(&d.y_test.data[i * y_onehot->cols], &y_onehot->data[idx[n_train + i] * y_onehot->cols], y_onehot->cols * sizeof(float));
    }
    free(idx);
    return d;
}

void dl_free_dataset(Dataset *d)
{
    mat_free(&d->X_train); mat_free(&d->X_test);
    mat_free(&d->y_train); mat_free(&d->y_test);
}

// NETWORK CREATION

static float randn_small(void)
{
    // Box-Muller (sigma=0.01)
    float u1 = ((float)rand() + 1.f) / ((float)RAND_MAX + 1.f);
    float u2 = ((float)rand() + 1.f) / ((float)RAND_MAX + 1.f);
    return 0.01f * sqrtf(-2.f * logf(u1)) * cosf(2.f * 3.14159265f * u2);
}

Network *dl_create_network(const NetConfig *cfg, int input_size)
{
    srand(cfg->seed);
    Network *net = (Network *)malloc(sizeof(Network));
    net->n_layers = cfg->n_layers;
    net->lr = cfg->learning_rate;
    net->layers = (Layer *)malloc(cfg->n_layers * sizeof(Layer));

    int in_size = input_size;
    for (int l = 0; l < cfg->n_layers; l++) {
        int out_size = cfg->layer_sizes[l];
        Layer *lay = &net->layers[l];

        // Kaiming init
        float scale = sqrtf(2.0f / (float)in_size);

        lay->W = mat_alloc(in_size, out_size);
        lay->b = mat_alloc(1, out_size);
        lay->dW = mat_alloc(in_size, out_size);
        lay->db = mat_alloc(1, out_size);
        lay->Z = mat_alloc(1, out_size);
        lay->A = mat_alloc(1, out_size);

        for (size_t i = 0; i < (size_t)(in_size * out_size); i++)
            lay->W.data[i] = randn_small() * scale / 0.01f;

        mat_zero(&lay->b);

        // Last layer always softmax; others follow developer spec
        if (l == cfg->n_layers - 1)
            lay->act = ACT_SOFTMAX;
        else
            lay->act = cfg->activations[l];

        lay->prelu_alpha = cfg->prelu_alpha;
        in_size = out_size;
    }
    return net;
}

void dl_free_network(Network *net)
{
    for (int l = 0; l < net->n_layers; l++) {
        Layer *lay = &net->layers[l];
        mat_free(&lay->W);  mat_free(&lay->b);
        mat_free(&lay->dW); mat_free(&lay->db);
        mat_free(&lay->Z);  mat_free(&lay->A);
    }
    free(net->layers);
    free(net);
}

// ACTIVATION FUNCTIONS

static void apply_activation(Matrix *m, Activation act, float alpha)
{
    size_t n = m->rows * m->cols;
    switch (act) {
    case ACT_PRELU:
        for (size_t i = 0; i < n; i++)
            m->data[i] = m->data[i] >= 0.f ? m->data[i] : alpha * m->data[i];
        break;
    case ACT_SIGMOID:
        for (size_t i = 0; i < n; i++)
            m->data[i] = 1.f / (1.f + expf(-m->data[i]));
        break;
    case ACT_TANH:
        for (size_t i = 0; i < n; i++)
            m->data[i] = tanhf(m->data[i]);
        break;
    case ACT_SOFTMAX:
        for (size_t r = 0; r < m->rows; r++) {
            float *row = m->data + r * m->cols;
            float  mx  = row[0];
            for (size_t c = 1; c < m->cols; c++) if (row[c] > mx) mx = row[c];
            float sum  = 0.f;
            for (size_t c = 0; c < m->cols; c++) { row[c] = expf(row[c] - mx); sum += row[c]; }
            for (size_t c = 0; c < m->cols; c++) row[c] /= sum;
        }
        break;
    }
}

// Derivative of activation w.r.t. Z, given A; written into dZ (same shape as Z/A)
static void activation_deriv(Matrix *dZ, const Matrix *A, Activation act, float alpha)
{
    size_t n = A->rows * A->cols;
    switch (act) {
    case ACT_PRELU:
        // dA/dZ = 1 if Z>0 else alpha -> A=prelu(Z) was stored, recover sign from A if A>0 -> Z>0
        for (size_t i = 0; i < n; i++)
            dZ->data[i] *= (A->data[i] > 0.f) ? 1.f : alpha;
        break;
    case ACT_SIGMOID:
        for (size_t i = 0; i < n; i++)
            dZ->data[i] *= A->data[i] * (1.f - A->data[i]);
        break;
    case ACT_TANH:
        for (size_t i = 0; i < n; i++)
            dZ->data[i] *= 1.f - A->data[i] * A->data[i];
        break;
    case ACT_SOFTMAX:
        // Combined with cross-entropy: gradient is simply (A - y), handled in backprop
        break;
    }
}

//FORWARD PASS

// Resize Z and A of each layer to accommodate batch rows
static void resize_layer_cache(Layer *lay, size_t batch, size_t out)
{
    if (lay->Z.rows != batch || lay->Z.cols != out) {
        mat_free(&lay->Z); mat_free(&lay->A);
        lay->Z = mat_alloc(batch, out);
        lay->A = mat_alloc(batch, out);
    }
}

static void forward_layer(Layer *lay, const Matrix *in_A)
{
    size_t batch = in_A->rows;
    size_t out = lay->W.cols;
    resize_layer_cache(lay, batch, out);

    mat_mul(&lay->Z, in_A, &lay->W);

    for (size_t i = 0; i < batch; i++)
        for (size_t j = 0; j < out; j++)
            lay->Z.data[i * out + j] += lay->b.data[j];

    mat_copy(&lay->A, &lay->Z);
    apply_activation(&lay->A, lay->act, lay->prelu_alpha);
}

static void forward_pass(Network *net, const Matrix *X)
{
    const Matrix *in = X;
    for (int l = 0; l < net->n_layers; l++) {
        forward_layer(&net->layers[l], in);
        in = &net->layers[l].A;
    }
}

// LOSS

float dl_cross_entropy_loss(const Matrix *probs, const Matrix *y_onehot)
{
    float loss = 0.f;
    size_t n = probs->rows;
    size_t c = probs->cols;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < c; j++)
            if (y_onehot->data[i * c + j] > 0.f) {
                float p = probs->data[i * c + j];
                loss -= logf(p + 1e-9f);
            }
    return loss / (float)n;
}

// BACKWARD PASS + PARAMETER UPDATE

static void backward_pass(Network *net, const Matrix *X_batch, const Matrix *y_batch)
{
    int L = net->n_layers;
    float lr = net->lr;
    size_t batch = X_batch->rows;
    float scale = 1.f / (float)batch;

    // dA for the current layer (starts at output layer)
    Matrix delta = mat_alloc(batch, net->layers[L-1].A.cols);

    // Output layer: dZ = A - y (softmax + cross-entropy combined)
    {
        Layer *lay = &net->layers[L - 1];
        size_t out = lay->A.cols;
        for (size_t i = 0; i < batch * out; i++)
            delta.data[i] = (lay->A.data[i] - y_batch->data[i]) * scale;
    }

    for (int l = L - 1; l >= 0; l--) {
        Layer *lay = &net->layers[l];
        const Matrix *A_prev = (l == 0) ? X_batch : &net->layers[l-1].A;
        size_t in_sz = A_prev->cols;
        size_t out_sz = lay->A.cols;

        // dW = A_prev^T * delta
        mat_mul_atb(&lay->dW, A_prev, &delta);

        // db = sum of delta rows
        mat_zero(&lay->db);
        for (size_t i = 0; i < batch; i++)
            for (size_t j = 0; j < out_sz; j++)
                lay->db.data[j] += delta.data[i * out_sz + j] * scale;

        // Propagate delta to previous layer (unless currently at layer 0)
        if (l > 0) {
            Matrix new_delta = mat_alloc(batch, in_sz);
            mat_mul_abt(&new_delta, &delta, &lay->W);
            activation_deriv(&new_delta, &net->layers[l-1].A, net->layers[l-1].act, net->layers[l-1].prelu_alpha);
            mat_free(&delta);
            delta = new_delta;
        }

        // Update parameters: W -= lr * dW,  b -= lr * db
        for (size_t i = 0; i < in_sz * out_sz; i++)
            lay->W.data[i] -= lr * lay->dW.data[i];
        for (size_t j = 0; j < out_sz; j++)
            lay->b.data[j] -= lr * lay->db.data[j];
    }
    mat_free(&delta);
}

// TRAIN

void dl_train(Network *net, const NetConfig *cfg, const Matrix *X, const Matrix *y_onehot)
{
    size_t n = X->rows;
    size_t feat = X->cols;
    size_t cls = y_onehot->cols;
    int batch_size = cfg->batch_size;

    // Index shuffle array
    size_t *idx = (size_t *)malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) idx[i] = i;

    // Temporary batch matrices
    Matrix X_batch = mat_alloc(batch_size, feat);
    Matrix y_batch = mat_alloc(batch_size, cls);

    for (int epoch = 0; epoch < cfg->epochs; epoch++) {
        // Shuffle
        for (size_t i = n - 1; i > 0; i--) {
            size_t j = (size_t)rand() % (i + 1);
            size_t tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }

        float epoch_loss = 0.f;
        int n_batches = 0;

        for (size_t start = 0; start + batch_size <= n; start += batch_size) {
            // Fill batch
            for (int b = 0; b < batch_size; b++) {
                size_t si = idx[start + b];
                memcpy(&X_batch.data[b * feat], &X->data[si * feat], feat * sizeof(float));
                memcpy(&y_batch.data[b * cls], &y_onehot->data[si * cls], cls * sizeof(float));
            }

            forward_pass(net, &X_batch);

            // Accumulate loss from output layer probabilities
            epoch_loss += dl_cross_entropy_loss(&net->layers[net->n_layers-1].A, &y_batch);
            n_batches++;

            backward_pass(net, &X_batch, &y_batch);
        }

        epoch_loss /= (float)n_batches;
        float acc = dl_evaluate(net, X, y_onehot);
        printf("Epoch %4d/%d  |  loss: %.6f  |  accuracy: %.4f\n", epoch + 1, cfg->epochs, epoch_loss, acc);
    }

    mat_free(&X_batch);
    mat_free(&y_batch);
    free(idx);
}

// PREDICT & EVALUATE

void dl_predict(Network *net, const Matrix *X, Matrix *out_probs)
{
    forward_pass(net, X);
    mat_copy(out_probs, &net->layers[net->n_layers - 1].A);
}

float dl_evaluate(Network *net, const Matrix *X, const Matrix *y_onehot)
{
    // Running in chunks to avoid huge memory allocation
    size_t n = X->rows;
    size_t feat = X->cols;
    size_t cls = y_onehot->cols;
    int chunk = 256;
    int correct = 0;

    Matrix Xc = mat_alloc(chunk, feat);
    Matrix Ac = mat_alloc(chunk, cls);

    for (size_t start = 0; start < n; start += chunk) {
        size_t end = start + chunk;
        if (end > n) end = n;
        size_t cur = end - start;

        memcpy(Xc.data, &X->data[start * feat], cur * feat * sizeof(float));
        Xc.rows = cur;
        forward_pass(net, &Xc);
        Layer *out = &net->layers[net->n_layers - 1];

        for (size_t i = 0; i < cur; i++) {
            // Predicted class = argmax of probs
            size_t pred = 0, truth = 0;
            for (size_t j = 1; j < cls; j++) {
                if (out->A.data[i * cls + j] > out->A.data[i * cls + pred])
                    pred = j;
                if (y_onehot->data[(start + i) * cls + j] >
                    y_onehot->data[(start + i) * cls + truth])
                    truth = j;
            }
            if (pred == truth) correct++;
        }
    }
    Xc.rows = chunk;
    mat_free(&Xc);
    mat_free(&Ac);
    return (float)correct / (float)n;
}
