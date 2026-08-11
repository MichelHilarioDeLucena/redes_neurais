#include "nnet.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#define FWR_ACTIV_DROPOUT(A_FUNC)                                              \
  for (; pout < end; pout++, pin++, pmask++) {                                 \
    float alive = xorshift_float(net->rstate);                                 \
    *pout = A_FUNC(*pin);                                                      \
    if (alive < net->p_alive)                                                  \
      *pmask = 1.f, *pout *= scale;                                            \
    else                                                                       \
      *pmask = 0.f, *pout = 0.f;                                               \
  }

#define FWR_ACTIV(A_FUNC)                                                      \
  for (; pout < end; pout++, pin++, pmask++) {                                 \
    float alive = xorshift_float(net->rstate);                                 \
    *pout = A_FUNC(*pin);                                                      \
  }

nnet *create_nnet(scheme_nn scheme[],thread_pool* tp, params_nnet *param) {

  nnet *net = malloc(sizeof(nnet));

  *net = (nnet){.n_layers   = param->n_layers,
                .batch_size = param->batch_size,
                .input_size = param->input_size,
                .t_step     = 1,
                .b1 	      = param->b1,
                .b2 	      = param->b2,
                .lambda     = param->lambda,
                .p_alive    = param->p_alive,
                .off_dropout= param->off_dropout,
                .rstate     = param->rstate,
                .learn_rt   = param->learn_rt};
                
  uint32_t n_layers=param->n_layers;
  uint32_t b_size=param->batch_size;
  net->layers=calloc(n_layers,sizeof(nnet_layer));
  nnet_layer *nn_l = net->layers;
  if(!param->input_data)
    net->layers->in=new_matrix(param->batch_size,scheme[0].input_size);
  else
    net->layers->in=new_matrix_set_data(
      param->batch_size,scheme[0].input_size,param->input_data);

    
  nn_l->grad_in=NULL;
  
  
  for (uint32_t l = 0; l < n_layers; l++) {
    nn_l = net->layers + l;
    nn_l->t_layer = scheme[l].type;
    if (l > 0) {
      nn_l->in = (nn_l - 1)->out;
      nn_l->grad_in = (nn_l - 1)->grad_out;
    }
    nn_l->t_in = new_matrix(nn_l->in->col,nn_l->in->row);
    switch(nn_l->t_layer){
    case DENSE:{
      dense_layer *dense=&nn_l->type.dense;
      uint32_t hsize=scheme[l].tag.dense.hidden_size;
      uint32_t isize = nn_l->in->col;

      dense->a_func   = scheme[l].tag.dense.activ;
      dense->W        = new_matrix(isize, hsize);
      dense->tW       = new_matrix(hsize, isize);
      dense->mW       = new_matrix(isize, hsize);
      dense->vW       = new_matrix(isize, hsize);
      dense->dW       = new_matrix(isize, hsize);
      dense->bias     = new_matrix(1, hsize);
      dense->dB       = new_matrix(1, hsize);
      dense->mB       = new_matrix(1, hsize);
      dense->vB       = new_matrix(1, hsize);
      dense->Z        = new_matrix(b_size, hsize);
      dense->mask     = new_matrix(b_size, hsize);
      nn_l->out       = new_matrix(b_size, hsize);
      nn_l->grad_out  = new_matrix(b_size, hsize);

      if(param->rstate){
        if (dense->a_func == RELU || dense->a_func == L_RELU)
          init_uniform_distr_He_xors64(dense->W, isize,net->rstate);
        else
          init_uniform_distr_xors64(dense->W, isize, hsize,net->rstate);
      }else{
        if (dense->a_func == RELU || dense->a_func == L_RELU)
          init_uniform_distr_He(dense->W, isize);
        else
          init_uniform_distr(dense->W, isize, hsize);
      }
      transpose_by(dense->W,dense->tW);
    }break;
    case DENSE_LINEAR:{
      linear_layer *linear=&nn_l->type.linear;
      
      uint32_t hsize=scheme[l].tag.linear.hidden_size;
      uint32_t isize = nn_l->in->col;
      linear->W        = new_matrix(isize, hsize);
      linear->tW       = new_matrix(hsize, isize);
      linear->mW       = new_matrix(isize, hsize);
      linear->vW       = new_matrix(isize, hsize);
      linear->dW       = new_matrix(isize, hsize);
      if(scheme[l].tag.linear.use_bias){
        linear->bias     = new_matrix(1, hsize);
        linear->dB       = new_matrix(1, hsize);
        linear->mB       = new_matrix(1, hsize);
        linear->vB       = new_matrix(1, hsize);
      }
      linear->Z        = new_matrix(b_size, hsize);
      nn_l->out       = new_matrix(b_size, hsize);
      nn_l->grad_out  = new_matrix(b_size, hsize);
      if(param->rstate)
          init_uniform_distr_He_xors64(linear->W, isize,net->rstate);
      else
          init_uniform_distr_He(linear->W, isize);
      
      transpose_by(linear->W,linear->tW);
    }break;
    case ACTIVATION:{
      activation_layer *activ=&nn_l->type.activ;
      activ->a_func   = scheme[l].tag.activ_l.activ;
      activ->mask     = new_matrix(b_size, nn_l->in->col);
      nn_l->out       = new_matrix(b_size, nn_l->in->col);
      nn_l->grad_out  = new_matrix(b_size, nn_l->in->col);
    }break;
    case BATCH_NORM:{
      b_norm_layer *bnorm=&nn_l->type.bnorm;
      uint32_t osize   = nn_l->in->col;
      bnorm->momentum=.99f;
      bnorm->epsilon=1e-5f;
      bnorm->W_norm    = new_matrix(1,osize);
      bnorm->bias_norm = new_matrix(1,osize);
      bnorm->dW_norm   = new_matrix(1,osize);
      bnorm->dB_norm   = new_matrix(1,osize);
      bnorm->run_mean  = new_matrix(1,osize);
      bnorm->run_var   = new_matrix(1,osize);
      bnorm->mean_bf   = new_matrix(1,osize);
      bnorm->var_bf    = new_matrix(1,osize);
      bnorm->std_inv   = new_matrix(1,osize);
      bnorm->mW        = new_matrix(1, osize);
      bnorm->mB        = new_matrix(1, osize);
      bnorm->vW        = new_matrix(1, osize);
      bnorm->vB        = new_matrix(1, osize);
      bnorm->x_hat     = new_matrix(b_size, osize);
      bnorm->dx_hat    = new_matrix(b_size, osize);
      bnorm->sum_dx_hat= new_matrix(1, osize);
      bnorm->sum_dxx   = new_matrix(1, osize);
      nn_l->out        = new_matrix(b_size, osize);
      nn_l->grad_out   = new_matrix(b_size, osize);
      float *p = bnorm->W_norm->data;
      float *end = bnorm->W_norm->end;
      while (p < end) *p++ = 1.0f;
    }
    }
  }
  
  net->tp =!tp ? new_thread_pool() : tp;
  return net;
}

void forward_pass(nnet *net, STATE_RUN state) {

  for (int l = 0; l < net->n_layers; l++) {
    nnet_layer *nn_l = net->layers + l;
    switch (nn_l->t_layer) {
    case DENSE: {
      dense_layer *dense = &nn_l->type.dense;
      uint32_t apply_dropout = (state & TRAIN) && !(state & OFF_DROPOUT);
      threaded_matmult(nn_l->in, dense->W, dense->Z, NN, true, net->tp);
      threaded_forward(dense->bias, dense->mask, nn_l->out, dense->Z,
                       net->p_alive, apply_dropout, dense->a_func, net->tp);
    } break;

    case DENSE_LINEAR: {
      linear_layer *linear = &nn_l->type.linear;
      threaded_matmult(nn_l->in, linear->W, nn_l->out, NN, true, net->tp);
      if(linear->bias)
        matrix_sum_broadcast(nn_l->out,linear->bias);
    } break;
    case ACTIVATION: {
      activation_layer *activ = &nn_l->type.activ;
      uint32_t apply_dropout = (state & TRAIN) && !(state & net->off_dropout);
      float *pin = nn_l->in->data;
      float *pout = nn_l->out->data;
      float *end = nn_l->out->end;
      float *pmask = activ->mask->data;
      float scale = 1.f / net->p_alive;
      if(activ->a_func==LOG_SOFTMAX){
        memcpy(nn_l->out->data, nn_l->in->data, nn_l->in->len * sizeof(float));
        log_softmax(nn_l->out);
      }else if (apply_dropout)
        switch (activ->a_func) {
        case SIGMOID: {FWR_ACTIV_DROPOUT(sigmoid)} break;
        case TANH:    {FWR_ACTIV_DROPOUT(tanf)} break;
        case RELU:    {FWR_ACTIV_DROPOUT(ReLU)} break;
        case L_RELU:  {FWR_ACTIV_DROPOUT(leaky_ReLU)} break;
        case SILU:    {FWR_ACTIV_DROPOUT(silu)} break;
        }
      else {
        switch (activ->a_func) {
        case SIGMOID: {FWR_ACTIV(sigmoid)} break;
        case TANH:    {FWR_ACTIV(tanf)} break;
        case RELU:    {FWR_ACTIV(ReLU)} break;
        case L_RELU:  {FWR_ACTIV(leaky_ReLU)} break;
        case SILU:    {FWR_ACTIV(silu)} break;
        }
      }
    } break;
    case BATCH_NORM: {
      b_norm_layer *bnorm = &nn_l->type.bnorm;
      uint32_t M = net->batch_size;
      uint32_t N = nn_l->in->col;
      uint32_t len = M * N;
      float *in = nn_l->in->data;
      
      memset(bnorm->mean_bf->data, 0, N * sizeof(float));
      memset(bnorm->var_bf->data, 0, N * sizeof(float));
      if (state & TRAIN) {
        float over_b = 1.f / net->batch_size;
        float mmtun = bnorm->momentum;
        for (uint32_t i = 0; i < len; i += N)
          for (uint32_t j = 0; j < N; j++)
            bnorm->mean_bf->data[j] += in[i + j];
        for (uint32_t j = 0; j < N; j++)
          bnorm->mean_bf->data[j] *= over_b;

        for (uint32_t i = 0; i < len; i += N)
          for (uint32_t j = 0; j < N; j++) {
            float diff = in[i + j] - bnorm->mean_bf->data[j];
            bnorm->var_bf->data[j] += diff * diff;
          }
        for (uint32_t j = 0; j < N; j++)
          bnorm->var_bf->data[j] *= over_b;
        for (uint32_t j = 0; j < N; j++) {
          float mean = bnorm->mean_bf->data[j];
          float var = bnorm->var_bf->data[j];
          float r_mean = bnorm->run_mean->data[j];
          float r_var = bnorm->run_var->data[j];
          bnorm->run_mean->data[j] = r_mean * mmtun + (1 - mmtun) * mean;
          bnorm->run_var->data[j] = r_var * mmtun + (1 - mmtun) * var;
        }
      }
      float *mean_ptr, *var_ptr;
      float *x_hat = bnorm->x_hat->data;
      float *W = bnorm->W_norm->data;
      float *b = bnorm->bias_norm->data;
      float *r_mean = bnorm->run_mean->data;
      float *inv_std = bnorm->std_inv->data;

      if (state & TRAIN) {
        mean_ptr = bnorm->mean_bf->data;
        var_ptr = bnorm->var_bf->data;
      } else {
        mean_ptr = bnorm->run_mean->data;
        var_ptr = bnorm->run_var->data;
      }

      for (uint32_t j = 0; j < N; j++)
        bnorm->std_inv->data[j] = 1.f / sqrtf(var_ptr[j] + bnorm->epsilon);

      for (uint32_t i = 0; i < len; i += N)
        for (uint32_t j = 0; j < N; j++)
          x_hat[i + j] = (in[i + j] - mean_ptr[j]) * inv_std[j];

      for (uint32_t i = 0; i < len; i += N)
        for (uint32_t j = 0; j < N; j++)
          nn_l->out->data[i + j] = x_hat[i + j] * W[j] + b[j];
      break;
    }
    }
  }
}

void backprop(matrix *target, nnet *net) {  
  nnet_layer *nn_l;
  for (int l = net->n_layers-1; l >= 0; l--) {
    nn_l=net->layers+l;
    switch (nn_l->t_layer){
    case DENSE:{
      dense_layer *dense=&nn_l->type.dense;
      if(dense->a_func==LOG_SOFTMAX){
        matrix_exp_sub(nn_l->out, target, nn_l->grad_out);
        matrix_scalar_prod(nn_l->grad_out, 1.f / net->batch_size);
      }else{
        uint32_t apply_dropout = !(net->off_dropout & OFF_DROPOUT);
        threaded_backward(dense->Z, dense->mask, nn_l->grad_out,
           net->p_alive,apply_dropout,dense->a_func, net->tp);
      }
      if(nn_l->grad_in)
        threaded_matmult(nn_l->grad_out, dense->tW, nn_l->grad_in,NN,true,net->tp);
      transpose_by(nn_l->in,nn_l->t_in);
      threaded_matmult(nn_l->t_in, nn_l->grad_out, dense->dW, NN,true,net->tp);
      matrix *grad_out = nn_l->grad_out;
      matrix *dB = dense->dB;
      memset(dB->data,0,dB->len*sizeof(float));
        
      for (uint32_t r = 0; r < grad_out->len; r+=grad_out->col) 
        for (uint32_t c = 0; c < grad_out->col; c++) 
          dB->data[c] += grad_out->data[r + c];
    }break;
    case DENSE_LINEAR:{
    linear_layer *linear=&nn_l->type.linear;
    if(nn_l->grad_in)
      threaded_matmult(nn_l->grad_out, linear->tW, nn_l->grad_in,NN,true,net->tp);
    transpose_by(nn_l->in,nn_l->t_in);
    threaded_matmult(nn_l->t_in, nn_l->grad_out, linear->dW, NN,true,net->tp);
    if(linear->bias){

      matrix *grad_out = nn_l->grad_out;
      matrix *dB = linear->dB;
      memset(dB->data,0,dB->len*sizeof(float));
      
      for (uint32_t r = 0; r < grad_out->len; r+=grad_out->col) 
        for (uint32_t c = 0; c < grad_out->col; c++) 
          dB->data[c] += grad_out->data[r + c];  
    }
    }break;
       case ACTIVATION: {
    activation_layer *activ = &nn_l->type.activ;
    if (nn_l->grad_in) {
    memcpy(nn_l->grad_in->data, nn_l->grad_out->data, nn_l->grad_out->len * sizeof(float));
    uint32_t apply_dropout = !(net->off_dropout & OFF_DROPOUT);
    threaded_backward(nn_l->in, activ->mask, nn_l->grad_in,
                      net->p_alive, apply_dropout, activ->a_func, net->tp);
}
} break;
    case BATCH_NORM: {
      b_norm_layer *bnorm=&nn_l->type.bnorm;
      memset(bnorm->dW_norm->data, 0, bnorm->dW_norm->len * sizeof(float));
      memset(bnorm->dB_norm->data, 0, bnorm->dB_norm->len * sizeof(float));
      float *dout=nn_l->grad_out->data;
      float *x_hat=bnorm->x_hat->data;
      uint32_t M=nn_l->out->row;
      uint32_t N=nn_l->out->col, len = M*N;
      
      for (uint32_t i = 0; i < len; i+=N)
        for (uint32_t j = 0; j < N; j++)
          bnorm->dW_norm->data[j]+=dout[i+j]*x_hat[i+j],
          bnorm->dB_norm->data[j]+=dout[i+j];
      
      if(nn_l->grad_in){
        float *sum_dx_hat=bnorm->sum_dx_hat->data;
        float *sum_dxx=bnorm->sum_dxx->data;
        float one_invM = 1.0f / M;
        float *din=nn_l->grad_in->data;
        float *std_inv=bnorm->std_inv->data;
        float *dx_hat=bnorm->dx_hat->data;
        memset(bnorm->sum_dx_hat->data, 0, N * sizeof(float));
        memset(bnorm->sum_dxx->data, 0, N * sizeof(float));
        for (uint32_t i = 0; i < len; i+=N)
          for (uint32_t j = 0; j < N; j++)
            dx_hat[i+j]=dout[i+j] * bnorm->W_norm->data[j],
            sum_dx_hat[j]+=dx_hat[i+j],
            sum_dxx[j]+=dx_hat[i+j]*x_hat[i+j];
        
        for (uint32_t i = 0; i < len; i+=N)
          for (uint32_t j = 0; j < N; j++)
            din[i+j]=std_inv[j]*one_invM*(M*dx_hat[i+j]-sum_dx_hat[j]-x_hat[i+j]*sum_dxx[j]);
      }
    }break;
    }
  } 
}

void update_layers(nnet *net) {
  for (size_t l = 0; l < net->n_layers; l++) {
    nnet_layer *nn_l=net->layers+l;
    switch (nn_l->t_layer){
      case DENSE:{
        dense_layer *dense=&nn_l->type.dense;
        SGD(dense->W,dense->dW,net->learn_rt,5.f);
        SGD(dense->bias,dense->dB,net->learn_rt,5.f);
        transpose_by(dense->W, dense->tW);
      }break;
      case DENSE_LINEAR:{
        linear_layer *linear=&nn_l->type.linear;
        SGD(linear->W,linear->dW,net->learn_rt,5.f);
        if(linear->bias)
          SGD(linear->bias,linear->dB,net->learn_rt,5.f);
        transpose_by(linear->W, linear->tW);
      }break;
      case BATCH_NORM:{
        b_norm_layer *bnorm=&nn_l->type.bnorm;
        SGD(bnorm->W_norm,bnorm->dW_norm,net->learn_rt,5.f);
        SGD(bnorm->bias_norm,bnorm->dB_norm,net->learn_rt,5.f);
      }
    }
  }
}

void update_layers_adamw(nnet *net) {
  for (size_t l = 0; l < net->n_layers; l++) {
    nnet_layer *nn_l=net->layers+l;
    switch (nn_l->t_layer){
      case DENSE:{
        dense_layer *dense=&nn_l->type.dense;
        ADAMW_correction(dense->W, dense->mW, dense->vW,
          dense->dW, net->b1, net->b2, net->learn_rt,
          net->t_step, net->lambda);
        ADAMW_correction(dense->bias, dense->mB, dense->vB,
          dense->dB, net->b1, net->b2, net->learn_rt,
          net->t_step, 0.f);
          transpose_by(dense->W,dense->tW);
        }break;
      case DENSE_LINEAR:{
        linear_layer *linear=&nn_l->type.linear;
        ADAMW_correction(linear->W, linear->mW, linear->vW,
          linear->dW, net->b1, net->b2, net->learn_rt,
          net->t_step, net->lambda);
        if(linear->bias)
          ADAMW_correction(linear->bias, linear->mB, linear->vB,
            linear->dB, net->b1, net->b2, net->learn_rt,
            net->t_step, 0.f);        
        transpose_by(linear->W,linear->tW);
        }break;
        case BATCH_NORM:{
          b_norm_layer *bnorm=&nn_l->type.bnorm;
        ADAMW_correction(bnorm->W_norm, bnorm->mW, bnorm->vW,
          bnorm->dW_norm, net->b1, net->b2, net->learn_rt,
          net->t_step, net->lambda);
        ADAMW_correction(bnorm->bias_norm, bnorm->mB, bnorm->vB,
          bnorm->dB_norm, net->b1, net->b2, net->learn_rt,
          net->t_step, 0.f);          
        }        
      }
    }
  net->t_step++;
}

void run_nnet(size_t epoch_max, nnet *net, data_loader *dtl, STATE_RUN state,FILE *fout) {
  fputs("epoch,error,acc\n", fout);
  puts("\nstart\n");

  matrix *out_l=net->layers[net->n_layers - 1].out;
  uint32_t num_classes = out_l->col; 
  matrix *target = new_matrix(net->batch_size, num_classes);
  struct timespec t0, t1, delta;
  float error = 0, acc = 0, ba_md = (float)net->batch_size / dtl->size_lb;
  float lr_og = net->learn_rt;
  size_t end = dtl->n_itens / net->batch_size;
  for (size_t e = 0, i; e < epoch_max; e++) {
    error = 0;
    acc = 0;

    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (i = 0; i < end; i++) {
      load_batch(dtl, net->layers->in, target, i);
      struct timespec t2, t3, delta1;
      forward_pass(net, state);
      if (state & TRAIN) {
        backprop(target, net);
        // update_layers(net);
        update_layers_adamw(net);
      }
      error += cat_cross_entropy(out_l, target);
      acc += get_accuracy(out_l, target) * 100.0;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    // net->learn_rt -= lr_og * 0.1f;
    // if (e > 8)
    //   net->learn_rt = lr_og * .1f;
    // net->learn_rt*=0.99f;
    // if(e%5==0)net->learn_rt=lr_og;
    sub_timespec(t0, t1, &delta);
    printf("e = %ld | dt(s) = %d.%.4ld", e + 1, (int)delta.tv_sec,
           delta.tv_nsec);

    error *= ba_md;
    acc *= ba_md;
    printf(" loss = %.4f | acc = %.4f\n", error, acc);
    fprintf(fout, "%zu,%f,%f\n", e, error, acc);
    
    if(state==TRAIN)
      shuffle_data(dtl);
  }
  puts("\nend.\n");
  destroy_matrix(target);
}

void train_nnet(size_t epoch_max, nnet *net, data_loader *dtl, char *nmfile) {
  FILE *file_train = fopen(nmfile, "w");
  if (!file_train) {
    perror("\nERRO: nao foi possivel criar arquivo de treino.\n");
    exit(1);
  }
  run_nnet(epoch_max, net, dtl, TRAIN, file_train);
  fclose(file_train);
}

void out_nnet(nnet *net, data_loader *dtl, char *namef) {
  FILE *file_test = fopen(namef, "w");
  if (!file_test) {
    perror("\nERRO: nao foi possivel criar arquivo de teste.\n");
    exit(1);
  }
  run_nnet(1, net, dtl, TEST, file_test);
  fclose(file_test);
}
void nnet_load(nnet *net, const char *fpath) {
  FILE *file = fopen(fpath, "rb");
  if (file) {
    uint32_t len_l;
    if (!fread(&len_l, sizeof(uint32_t), 1, file))
      perror("falha ao ler.\n");
    for (int l = 0; l < len_l; l++) {
      nnet_layer *nn_l=net->layers+l;
      
      if(fread(&nn_l->t_layer, sizeof(uint32_t), 1, file)){
        perror("erro ao ler tipo de  camada.\n");
        break;
      }
      switch (nn_l->t_layer){
      case DENSE:{
        dense_layer *dense=&nn_l->type.dense;
        
        if(!fread(&dense->W->row, sizeof(uint32_t), 1, file)){
          perror("erro ao ler linhas da matriz de pesos.\n");
          break;
        }

        if(!fread(&dense->W->col, sizeof(uint32_t), 1, file)){
          perror("erro ao ler colunas da matriz de pesos.\n");
          break;
        }
        dense->W->len=dense->W->row*dense->W->col;
        if(!fread(dense->W->data, sizeof(float), dense->W->len,file)){
          perror("erro ao ler peso.\n");
          break;
        }
        transpose_by(dense->W, dense->tW);
        if( !fread(&dense->bias->len, sizeof(uint32_t), 1, file)  ||
            !fread(dense->bias->data, sizeof(float), dense->bias->len, file)){
          perror("erro ao ler vies.\n");
          break;
        }
    }break;
    case BATCH_NORM:{}break;
    }
  }
  } else {
    perror("ERRO ao carregar arquivo dos pesos.\n");
  }
  fclose(file);
  
}

void save_weights(nnet *net) {
  size_t t = time(0);
  char namefile[64];
  sprintf(namefile, "net_weights%zu.bin", t);
  FILE *file = fopen(namefile, "wb");
  if (file) {
    uint32_t len_l = net->n_layers;
    fwrite(&len_l, sizeof(uint32_t), 1, file);
    for (int l = 0; l < len_l; l++) {
      nnet_layer *nn_l=net->layers+l;
      fwrite(&nn_l->t_layer, sizeof(uint32_t), 1, file);
      switch (nn_l->t_layer){
      case DENSE:{
        dense_layer *dense=&nn_l->type.dense;
        fwrite(&dense->W->row, sizeof(uint32_t), 1, file);
        fwrite(&dense->W->col, sizeof(uint32_t), 1, file);
        fwrite(dense->W->data, sizeof(float), dense->W->len, file);
        fwrite(&dense->bias->len, sizeof(uint32_t), 1, file);
        fwrite(dense->bias->data, sizeof(float), dense->bias->len, file);
      }break;
      case BATCH_NORM:{}
    }
    }
  } else
    perror("ERRO ao salvar pesos.\n");
  fclose(file);
}

void destroy_nnet(nnet *net) {
  for (int l = 0; l < net->n_layers - 1; l++) {
    nnet_layer *nn_l=net->layers + l;
    free(nn_l->t_in);
    switch (nn_l->t_layer){
    case DENSE:{
      dense_layer *dense=&nn_l->type.dense;
      destroy_matrix(dense->W);
      destroy_matrix(dense->dW);
      destroy_matrix(dense->bias);
      destroy_matrix(dense->dB);
      destroy_matrix(dense->Z);
      destroy_matrix(dense->mask);
      destroy_matrix(dense->mB);
      destroy_matrix(dense->vB);
      destroy_matrix(dense->mW);
      destroy_matrix(dense->vW);
    }break;
    case BATCH_NORM:{

    }
    }
  }
  free(net->layers);

  
  destroy_thread_pool(net->tp);
  free(net);
}
