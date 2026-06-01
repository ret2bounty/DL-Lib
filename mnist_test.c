#include <stdio.h>
#include <stdlib.h>

#include "deeplearn.h"
#include "mnist_loader.h"

int main(void)
{
    Matrix X_all, y_all;
    int n = mnist_load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", &X_all, &y_all, 0);
    if (n < 0) return 1;

    Dataset ds = dl_split_dataset(&X_all, &y_all, 0.9f, 42);

    mat_free(&X_all);
    mat_free(&y_all);

    printf("Train: %zu  |  Val: %zu\n", ds.X_train.rows, ds.X_test.rows);

    Matrix X_test_mnist, y_test_mnist;
    mnist_load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", &X_test_mnist, &y_test_mnist, 0);

    // Network Config
    int layer_sizes[]  = { 256, 128, 10 };
    Activation activations[]  = { ACT_PRELU, ACT_TANH };

    NetConfig cfg = {
        .n_layers      = 3,
        .layer_sizes   = layer_sizes,
        .activations   = activations,
        .prelu_alpha   = 0.01f,
        .learning_rate = 0.01f,
        .epochs        = 20,
        .batch_size    = 64,
        .seed          = 1234
    };

    Network *net = dl_create_network(&cfg, 784);

    printf("\n=== Training ===\n");
    dl_train(net, &cfg, &ds.X_train, &ds.y_train);

    float val_acc = dl_evaluate(net, &ds.X_test, &ds.y_test);
    printf("\nValidation accuracy : %.4f  (%.2f%%)\n", val_acc, val_acc * 100.f);

    float test_acc = dl_evaluate(net, &X_test_mnist, &y_test_mnist);
    printf("MNIST test accuracy : %.4f  (%.2f%%)\n", test_acc, test_acc * 100.f);

    dl_free_network(net);
    dl_free_dataset(&ds);
    mat_free(&X_test_mnist);
    mat_free(&y_test_mnist);

    return 0;
}
