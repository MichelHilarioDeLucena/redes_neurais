#include "data_loader/mnist_driver.h"
#include "RNN/rnnet.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main() {
  const char *train_path_labels = "archive_MNIST/train-labels.idx1-ubyte";
  const char *train_path_images = "archive_MNIST/train-images.idx3-ubyte";
  const char *test_path_labels  = "archive_MNIST/t10k-labels.idx1-ubyte";
  const char *test_path_images  = "archive_MNIST/t10k-images.idx3-ubyte";

  uint32_t batch = 36, t_step = 28, epochs = 5;
  float lr = 0.02;
  srand(time(0));
  scheme_rnn scheme[] = {{
      .activ = TANH,
      .input_size = 28,
      .hidden_size = 64,
      .out_size = 10,
  }};

  size_t n_ly = sizeof(scheme) / sizeof(scheme[0]);
  rnnet *rnn = create_rnnet(scheme, batch, n_ly, lr, t_step);

  printf("Carregando dataset de treino...\n");
  data_loader *train_ld = mnist_load(train_path_images, train_path_labels);

  train_nnet(epochs, rnn, train_ld, "rnn_train.csv");

  destroy_loader(train_ld);

  printf("Carregando dataset de teste...\n");
  data_loader *test_ld = mnist_load(test_path_images, test_path_labels);

  out_nnet(rnn, test_ld, "rnn_test.csv");

  destroy_loader(test_ld);
  return 0;
}
