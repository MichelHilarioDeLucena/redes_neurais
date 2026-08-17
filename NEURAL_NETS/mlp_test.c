#include "MLP/nnet.h"
#include "data_loader/data_loader.h"
#include "data_loader/mnist_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
  size_t t = time(0);
  srand(t);

  const char *train_path_labels = "archive_MNIST/train-labels.idx1-ubyte";
  const char *train_path_images = "archive_MNIST/train-images.idx3-ubyte";

  const char *test_path_labels = "archive_MNIST/t10k-labels.idx1-ubyte";
  const char *test_path_images = "archive_MNIST/t10k-images.idx3-ubyte";

  data_loader *data_ld = mnist_load(train_path_images, train_path_labels);

  

  XorShift64State rstate = {.val = 982173757522LL + time(0)};

  scheme_nn scheme[] = {
		{	.input_size = data_ld->stride_in,
			.type=DENSE_LINEAR,
			.tag.linear={.hidden_size=120} },
		{	.type=BATCH_NORM_MLP },
		{	.type=ACTIV_MLP,.tag.activ_l={.activ=RELU} },

		{	.type=DENSE_LINEAR,
			.tag.linear={.hidden_size=120} },
		{	.type=BATCH_NORM_MLP },
		{	.type=ACTIV_MLP,.tag.activ_l={.activ=RELU} },
    
    { .type=DENSE,
      .tag.dense={.activ= LOG_SOFTMAX,.hidden_size=10}}
	};
  // scheme_nn scheme[] = {
	// 	{	.input_size = data_ld->stride_in,
	// 		.type=DENSE,
	// 		.tag.dense={.activ=RELU,.hidden_size=120} },
	// 	{ .type=DENSE,
	// 		.tag.dense={.activ=RELU,.hidden_size=120} },		
  //   { .type=DENSE,
  //     .tag.dense={.activ= LOG_SOFTMAX,.hidden_size=10}}
	// };
	size_t batch_size=80, epochs=15;
  params_nnet params = {
      .off_dropout=OFF_DROPOUT,
      .b1 = .99f,
      .b2 = .999f,
      .batch_size = batch_size,
      .input_size = data_ld->stride_in,
      .lambda = .0f,
      .p_alive = 1.f,
      .rstate = &rstate,
      .t_step = 1,
      .n_layers = sizeof(scheme)/sizeof(*scheme),
      .learn_rt = .001f,
      .input_data=NULL
  };
  nnet *mnist_net = create_nnet(scheme, NULL,&params);

  train_nnet(epochs, mnist_net, data_ld, "mlp_train.csv");
  // save_weights(mnist_net);
  destroy_loader(data_ld);

  // nnet_load(mnist_net,"net_weights1776885940.bin");
  data_ld = mnist_load(test_path_images, test_path_labels);

  // set_batch(mnist_net,100);
  out_nnet(mnist_net, data_ld, "mlp_test.csv");

  destroy_loader(data_ld);
  destroy_nnet(mnist_net);

  return 0;
}
