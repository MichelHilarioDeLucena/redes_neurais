#ifndef NNETWORK
#define NNETWORK

#define RAND_RANGE 1000000

#include "../mmath_lib/mm_lib.h"
#include "../mmath_lib/mm_threaded.h"
#include "../data_loader/data_loader.h"
#include <stdio.h>

typedef enum STATE_RUN {
  TEST,
  TRAIN,
  FORWARD,
} STATE_RUN;

typedef struct nnet {
  uint32_t n_layers, batch_size, input_size;
  uint64_t t_step;
  float b1, b2, lambda, p_alive;
  matrix **weights;
  matrix **t_weights;
  matrix **bias;
  matrix **outputs;
  matrix **t_outputs;

  matrix **deltas;
  matrix **z_out;
  matrix **gw_buffer;
  matrix **gb_buffer;

  matrix **mw_adam;
  matrix **vw_adam;
  matrix **mb_adam;
  matrix **vb_adam;
  XorShift64State *rand_state;
  matrix **mask;
  matrix *input_grad;
  float learn_rt;
  thread_pool *tp;
  AFUNC_TYPE a_func;
} nnet;

nnet *create_nnet(uint32_t scheme[], uint32_t count_lyr, uint32_t bsize,
                  uint32_t in_size, AFUNC_TYPE func, thread_pool* tp,
                  float eta,float lambda,float p_alive,XorShift64State *state);
void forward_pass(matrix *input, nnet *net, STATE_RUN state);
void backprop(matrix *target, nnet *net);
void update_layers(nnet *net);
void update_layers_adamw(nnet *net);

void run_nnet(size_t epoch_max, nnet *net, data_loader *dtl, STATE_RUN state,
              FILE *file);
void train_nnet(size_t epoch_max, nnet *net, data_loader *dtl, char *namef);
void out_nnet(nnet *net, data_loader *dtl, char *namef);

void nnet_load(nnet *net, const char *path);
void save_weights(nnet *net);

void destroy_nnet(nnet *net);

#endif
