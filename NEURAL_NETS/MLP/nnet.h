#ifndef NNETWORK
#define NNETWORK

#include "../mmath_lib/mm_lib.h"
#include "../mmath_lib/mm_threaded.h"
#include "../data_loader/data_loader.h"
#include <stdio.h>

typedef enum STATE_RUN {
  TEST        =    0b1,
  TRAIN       =   0b10,
  FORWARD     =  0b100,
  OFF_DROPOUT = 0b1000,
} STATE_RUN;

typedef enum nn_layer_t { BATCH_NORM, DENSE ,DENSE_LINEAR,ACTIVATION} nn_layer_t;

typedef struct scheme_nn{  
  nn_layer_t type;
  uint32_t input_size;
  
  union{    
    struct {
      uint32_t hidden_size;
      AFUNC_TYPE activ;
    }dense;
    struct {
      uint16_t hidden_size,use_bias;
    }linear;
    struct {
      AFUNC_TYPE activ;
    }activ_l;
    struct {uint32_t size;}bnorm;
  }tag;
}scheme_nn;

typedef struct dense_layer{
  matrix *Z,*W,*bias;
  matrix *mask;
  matrix *tW;
  
  matrix *dW, *dB;
  
  matrix *mW, *vW;
  matrix *mB, *vB;
  AFUNC_TYPE a_func;
}dense_layer;
typedef struct linear_layer{
  matrix *Z,*W,*bias;
  matrix *tW;
  
  matrix *dW, *dB;
  
  matrix *mW, *vW;
  matrix *mB, *vB;
}linear_layer;

typedef struct activation_layer{
  matrix *mask;
  AFUNC_TYPE a_func;
}activation_layer;

typedef struct b_norm_layer{
  float momentum,epsilon;
  matrix *W_norm, *bias_norm;
  matrix *dW_norm, *dB_norm;

  matrix *mW, *vW;
  matrix *mB, *vB;
  matrix *run_mean, *run_var;
  matrix *mean_bf, *var_bf;
  matrix *std_inv,*x_hat,*dx_hat,*sum_dx_hat,*sum_dxx;
}b_norm_layer;

typedef struct nnet_layer{
  nn_layer_t t_layer;
  matrix *in,*t_in;
  matrix *grad_in;
  matrix *out;
  matrix *grad_out;
  union nn_layer_union{
    dense_layer dense;
    linear_layer linear;
    activation_layer activ;
    b_norm_layer bnorm;
  }type;
  
}nnet_layer;

typedef struct params_nnet{
  float b1, b2, lambda, p_alive;
  STATE_RUN off_dropout;
  uint32_t n_layers, batch_size, input_size;  
  uint64_t t_step;
  float learn_rt;
  XorShift64State *rstate;
  float *input_data;
}params_nnet;

typedef struct nnet {
  STATE_RUN off_dropout;
  float b1, b2, lambda, p_alive;
  uint32_t n_layers, batch_size, input_size;  
  uint64_t t_step;
  float learn_rt;
  XorShift64State *rstate;
  nnet_layer *layers;
  thread_pool *tp;

} nnet;

nnet *create_nnet(scheme_nn scheme[], thread_pool* tp,params_nnet *params_nn);
void forward_pass(nnet *net, STATE_RUN state);
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
