#include "../data_loader/mnist_driver.h"
#include "grnnet.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main() {
  const char *train_path_labels = "../archive_MNIST/train-labels.idx1-ubyte";
  const char *train_path_images = "../archive_MNIST/train-images.idx3-ubyte";
  const char *test_path_labels = "../archive_MNIST/t10k-labels.idx1-ubyte";
  const char *test_path_images = "../archive_MNIST/t10k-images.idx3-ubyte";

  uint32_t batch = 72, t_step = 28, epochs = 5;
  float lr = 0.05;
  srand(time(0));

  scheme_grnn scheme[] = {
    { .type=GRU,
      .input_size=28,
      .config.gru.hidden_size=64,
    },
    { .type=DENSE,
      .config.dense.output_size=10,
    }
  };
  
  size_t n_ly = sizeof(scheme) / sizeof(scheme[0]);
  printf(": %ld\n",n_ly);
  params_grrn params={
    .batch    =batch,
    .lr       =lr,
    .mode     =MANY_TO_ONE,
    .n_layers =n_ly,
    .t_step   =t_step,
  };
  grnnet *grnn = create_grnnet(scheme, &params);

  data_loader *train_ld = mnist_load(train_path_images, train_path_labels);

  train_grnnet(epochs, grnn, train_ld, "net_train.csv");

  destroy_loader(train_ld);

  data_loader *test_ld = mnist_load(test_path_images, test_path_labels);

  out_grnnet(grnn, test_ld, "net_test.csv");

  destroy_loader(test_ld);
  return 0;
}