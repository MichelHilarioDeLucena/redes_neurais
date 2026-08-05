#ifndef RNNET
#define RNNET

#include "../data_loader/data_loader.h"
#include "../mmath_lib/mm_threaded.h"


#include <stdio.h>

typedef enum STATE_RUN {
  TEST,
  TRAIN,
  FORWARD,
} STATE_RUN;

typedef struct scheme_rnn{
    uint32_t input_size,hidden_size,out_size;
    AFUNC_TYPE activ;
}scheme_rnn;

typedef struct rnnet_layer{
    matrix **in;
    matrix **h;
    matrix **z;
    matrix **dz;
    matrix **out;
    matrix **dout;
    matrix *y_h;
    matrix *W_ih;
    matrix *W_hh;
    matrix *W_oh;
    matrix *b_h;
    matrix *b_o;

    matrix *dW_ih;
    matrix *dW_hh;
    matrix *dW_oh;
    matrix *db_h;
    matrix *db_o;

    matrix **dh;
    AFUNC_TYPE activ;
}rnnet_layer;

typedef struct rnnet{
    uint32_t time_step,batch_size,n_layers;
    float learn_rt;
    rnnet_layer *layers;
    thread_pool *tp;
}rnnet;
rnnet *create_rnnet(scheme_rnn *scheme,uint32_t batch,uint32_t n_layers,float lr,uint32_t t_step);
void forward_rnnet(rnnet *rnn);
void backprop_rnnet(rnnet *rnn,matrix *label);
void update_rnnet(rnnet *rnn);

void run_nnet(size_t epoch_max, rnnet *rnn, data_loader *dtl, STATE_RUN state,
              FILE *file);
void train_nnet(size_t epoch_max, rnnet *net, data_loader *dtl, char *namef);
void out_nnet(rnnet *net, data_loader *dtl, char *namef);

#endif
