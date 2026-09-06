#ifndef GRNNET
#define GRNNET

#include "../data_loader/data_loader.h"
#include "../mmath_lib/mm_threaded.h"

#include <stdio.h>

typedef enum STATE_RUN {
  TEST,
  TRAIN,
  FORWARD,
} STATE_RUN;

typedef enum grnn_mode { MANY_TO_MANY=0b1, MANY_TO_ONE=0b10, GRAD_IN=0b100 } grnn_mode;
typedef enum grnn_layer_t { GRU, DENSE_GRNN } grnn_layer_t;

typedef struct scheme_grnn{
    grnn_layer_t type;
    uint32_t input_size;
    union {
        struct { uint32_t hidden_size;bool LN; } gru;
        struct { uint32_t output_size; } dense;
    } config;
} scheme_grnn;

typedef struct gate{
  float epsilon;
  matrix *Wh,*Wi, *b,*gamma;
  matrix *dWh,*dWi, *db,*dgamma;
  matrix *tWh,*tWi;
  matrix **out,**dout;
  matrix **mean, **var;
  matrix **std_inv,**x_hat;
}gate;

typedef struct gru_layer{
  bool use_layer_norm;
  matrix **h;
  matrix **dh;
  gate z,r,n;
} gru_layer;

typedef struct dense_layer{
  matrix *W, *b;
  matrix *dW, *db;
} dense_layer;

typedef struct grnnet_layer {
  grnn_layer_t t_layer;
  matrix **in;
  matrix **grad_in;
  matrix **out;
  matrix **grad_out;

  matrix *y1, *y2, *y3, *y4,*t_i,*t_g;

  union grrn_layer_union{
    gru_layer gru;
    dense_layer dense;
  } layer;
} grnnet_layer;

typedef struct params_grrn{
  uint32_t batch;
  uint32_t n_layers;
  uint32_t t_step;
  grnn_mode mode;
  float lr;
}params_grrn;

typedef struct grnnet {
  grnn_mode mode;
  uint32_t time_step, batch_size, n_layers;
  float learn_rt;
  grnnet_layer *layers;
  thread_pool *tp;
} grnnet;

grnnet *create_grnnet(scheme_grnn *scheme,params_grrn *params_grn);
void forward_grnnet(grnnet *rnn);
void backprop_grnnet(grnnet *rnn, matrix *label);
void update_grnnet(grnnet *rnn);

void run_grnnet(size_t epoch_max, grnnet *rnn, data_loader *dtl,
                STATE_RUN state, FILE *file);
void train_grnnet(size_t epoch_max, grnnet *net, data_loader *dtl, char *namef);
void out_grnnet(grnnet *grnet, data_loader *dtl, char *namef);
void init_gate(gate *g, uint32_t B, uint32_t H, uint32_t I,uint32_t T,bool use_ln);
void SGD_gate(gate *g,float lr,float maxg);
void reset_gate(gate *gate);
void destroy_gate(gate *gate,uint32_t T,bool use_ln);

void destroy_grnnet(grnnet *grnn);
#endif