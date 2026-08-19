#ifndef CNNET
#define CNNET
#include "../MLP/nnet.h"
#include <stdint.h>
#include <string.h>

typedef enum layer_cnn_t {
  CONV_LAYER   ,
  ACTIV_CNN    ,
  POOLING_LAYER,
  MLP_LAYER    ,
} layer_cnn_t;

typedef struct cnnet_params {
  uint32_t n_labels, in_N, in_H, in_W, in_C;
  float learn_rt, w_dec, pval;
  XorShift64State *rstate;
} cnnet_params;

typedef struct scheme_cnn {
  layer_cnn_t type;
  
  uint32_t stride, kh, kw;
  union {
    struct {
      uint32_t filters, padding;
    } conv;
    struct {
      activ_func activation;
    } activ;
    struct {
      uint32_t padd;
    } pool;
    struct {
      params_nnet *params_mlp;
      scheme_nn *scheme_mlp;
    } mlp;
  };
} scheme_cnn;

typedef struct conv_linear_l{
  matrix *W_mat, *dW_mat, *bias,*d_bias;
  matrix *mW_mat, *vW_mat,*mb_mat, *vb_mat;
  matrix *in_mat,*out_mat,*dZ_mat;
  matrix *t_W_mat,*t_in_mat;
  uint32_t filters, padding;
} conv_linear_l;

typedef struct activ_l_cnn{
  activ_func activation;
}activ_l_cnn;

typedef struct poolling_l{
  uint32_t *mask,padd;
} poolling_l;

typedef struct b_norm_layer_cnn{
  float momentum,epsilon;
  matrix *W_norm, *bias_norm;
  matrix *dW_norm, *dB_norm;

  matrix *mW, *vW;
  matrix *mB, *vB;
  matrix *run_mean, *run_var;
  matrix *mean_bf, *var_bf;
  matrix *std_inv,*x_hat,*dx_hat,*sum_dx_hat,*sum_dxx;
}b_norm_layer_cnn;

typedef struct cnnet_layer {
  tensor *in, *out;
  layer_cnn_t l_type;
  
  uint32_t stride, kh, kw;
  union {
    conv_linear_l conv;
    activ_l_cnn activ;
    poolling_l pool;
    b_norm_layer_cnn bnorm;
    nnet *mlp;
  } tag;
} cnnet_layer;

typedef struct cnnet {
  thread_pool *tp;
  XorShift64State *rstate;
  cnnet_layer *layers;
  nnet *mlp_head;
  uint32_t n_layers, batch_size, n_labels,t_step;
  float learn_rt,b1,b2;
} cnnet;

cnnet *create_cnnet(scheme_cnn *scheme,uint32_t n_layers,cnnet_params *params);
void forward_cnnet(cnnet *cnn,STATE_RUN state_run);
void backprop_cnnet(cnnet *cnn,matrix *labels);
void update_cnnet(cnnet *cnn);
void run_cnnet(size_t epoch_max, cnnet *cnet, data_loader *dtl, STATE_RUN state,
               FILE *fout);
void train_cnnet(size_t epoch_max, cnnet *cnet, data_loader *dtl,
                 char *nmfile);
void out_cnnet(cnnet *cnet, data_loader *dtl, char *namef);

void im2col(tensor *input, matrix *buffer, int k_w, int k_h, int stride,int padding);
void col2im(tensor *input, matrix *buffer, int k_w, int k_h, int stride,int padding);
void max_pooling(tensor *in, tensor *out, uint32_t *mask, uint32_t kh,
                 uint32_t kw, uint32_t stride, uint32_t padd);
void max_pooling_backward(tensor *dout, tensor *dinput, uint32_t *mask);
#endif
