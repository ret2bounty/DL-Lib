all:
	gcc -O3 -g -o mnist_test mnist_test.c mnist_loader.c deeplearn.c -lm
