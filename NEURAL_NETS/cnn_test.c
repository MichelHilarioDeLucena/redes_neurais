#include "data_loader/mnist_driver.h"
#include "CNN/cnnet.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
  size_t t = time(0);
  srand(t);
  XorShift64State rand={.val=time(0)+9383490184LL};
  uint32_t batch_size = 30, epochs = 5;
  uint32_t n_out = 10;
  float learn_rt = 0.002f;

  uint32_t mlp_layers[] = {0,120, 84, n_out};
  uint32_t mlp_l_count = sizeof(mlp_layers) / sizeof(*mlp_layers);
  
  scheme_cnn scheme[] = {
      {.type = CONV_LAYER,
       .activation = RELU,
       .kh = 5,
       .kw = 5,
       .stride = 1,
       .conv = {.filters = 6, .padding = 0}},

      {.type = POOLING_LAYER, .kh = 2, .kw = 2, .stride = 2},

      {.type = CONV_LAYER,
       .activation = RELU,
       .kh = 5,
       .kw = 5,
       .stride = 1,
       .conv = {.filters = 16, .padding = 0}},

      {.type = POOLING_LAYER, .kh = 2, .kw = 2, .stride = 2},

      {.type = DENSE_LAYER,
       .dense = {.layers = mlp_layers, .n_layers = mlp_l_count}}};

  uint32_t const N_LAYERS = sizeof(scheme) / sizeof(scheme_cnn);
  cnnet_params params = {.in_N = batch_size,
                         .in_H = 28,
                         .in_W = 28,
                         .in_C = 1,
                         .learn_rt = learn_rt,
                         .n_labels = n_out,
                         .pval = 1,
                         .w_dec = 0,
                        .rstate=&rand};
  cnnet *cnn = create_cnnet(scheme, N_LAYERS, &params);

  const char *train_path_labels = "archive_MNIST/train-labels.idx1-ubyte";
  const char *train_path_images = "archive_MNIST/train-images.idx3-ubyte";

  const char *test_path_labels = "archive_MNIST/t10k-labels.idx1-ubyte";
  const char *test_path_images = "archive_MNIST/t10k-images.idx3-ubyte";

  data_loader *data_ld = mnist_load(train_path_images, train_path_labels);

  train_cnnet(epochs, cnn, data_ld, "cnn_train.csv");
  destroy_loader(data_ld);

  data_ld = mnist_load(test_path_images, test_path_labels);

  out_cnnet(cnn, data_ld, "cnn_test.csv");
  destroy_loader(data_ld);
  

  return 0;
}
