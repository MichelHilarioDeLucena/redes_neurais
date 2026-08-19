#include "cnnet.h"
#include <math.h>
#include <stdlib.h>

cnnet *create_cnnet(scheme_cnn *scheme, uint32_t n_layers,
                    cnnet_params *params) {
  cnnet *cnn = calloc(1,sizeof(cnnet));
  cnn->layers = calloc(n_layers,sizeof(cnnet_layer));
  cnn->batch_size = params->in_N;
  cnn->n_layers = n_layers;
  cnn->learn_rt = params->learn_rt;
  cnn->n_labels = params->n_labels;
  cnn->rstate = params->rstate;
  cnn->t_step = 1;
  cnn->b1 = .9f;
  cnn->b2 = .999f;
  cnn->tp = new_thread_pool();

  cnn->layers[0].in = new_tensor_grad_init(params->in_N, params->in_H,
                                              params->in_W, params->in_C);

  for (uint32_t l = 0; l < n_layers; l++) {
    cnnet_layer *cnn_l=cnn->layers+l;
    cnn_l->l_type = scheme[l].type;
    
    cnn_l->stride = scheme[l].stride;
    cnn_l->kh = scheme[l].kh;
    cnn_l->kw = scheme[l].kw;

    if (l > 0)
      cnn_l->in = cnn->layers[l - 1].out;

    tensor *in = cnn_l->in;

    switch (cnn_l->l_type) {
    case CONV_LAYER: {
      conv_linear_l *conv=&cnn_l->tag.conv;
      uint32_t kh = scheme[l].kh;
      uint32_t kw = scheme[l].kw;
      uint32_t ho =
          (in->H - kh + 2 * scheme[l].conv.padding) / scheme[l].stride + 1;
      uint32_t wo =
          (in->W - kw + 2 * scheme[l].conv.padding) / scheme[l].stride + 1;
      uint32_t M = in->C * kw * kh;
      uint32_t N = ho * wo * params->in_N;
      uint32_t filters = scheme[l].conv.filters;
      conv->filters = filters;
      cnn_l->out = new_tensor_grad_init(params->in_N, ho, wo, filters);
      conv->padding = scheme[l].conv.padding;
      conv->in_mat = new_matrix(M, N);
      conv->W_mat = new_matrix(filters, M);
      conv->mW_mat = new_matrix(filters, M);
      conv->vW_mat = new_matrix(filters, M);
      conv->bias = new_matrix(filters, 1);
      conv->mb_mat = new_matrix(filters, 1);
      conv->vb_mat = new_matrix(filters, 1);
      conv->dW_mat = new_matrix(filters, M);
      conv->d_bias = new_matrix(filters, 1);
      conv->out_mat = new_matrix(filters, N);
      conv->dZ_mat = new_matrix(filters, N);
      conv->t_W_mat = new_matrix(M, filters);
      conv->t_in_mat = new_matrix(N, M);
      activ_func afunc=(cnn_l+1)->l_type==ACTIV_CNN?(cnn_l+1)->tag.activ.activation : LINEAR;
      uint32_t use_he=afunc == RELU || afunc == L_RELU || afunc == LINEAR;
      if (cnn->rstate) {
        if (use_he) init_uniform_distr_He_xors64(conv->W_mat, M,cnn->rstate);
        else init_uniform_distr_xors64(conv->W_mat, M, filters,cnn->rstate);
      } else {
        if (use_he) init_uniform_distr_He(conv->W_mat, M);
        else init_uniform_distr(conv->W_mat, M, filters);
      }
    }break;
    case ACTIV_CNN: {
      activ_l_cnn *activ=&cnn_l->tag.activ;
      activ->activation = scheme[l].activ.activation;
      cnn_l->out = new_tensor_grad_init(in->N,in->H, in->W, in->C);
    }break;
    case POOLING_LAYER: {
      poolling_l *pool=&cnn_l->tag.pool;
      uint32_t kh = scheme[l].kh, kw = scheme[l].kw;
      uint32_t ho = (in->H - kh) / scheme[l].stride + 1;
      uint32_t wo = (in->W - kw) / scheme[l].stride + 1;
      
      cnn_l->out = new_tensor_grad_init(params->in_N, ho, wo, in->C);
      pool->mask =calloc(cnn_l->out->len, sizeof(uint32_t));
    }break;
    case MLP_LAYER: {
      uint32_t input_size = in->H * in->W * in->C;
      scheme_nn *scheme_mlp= scheme[l].mlp.scheme_mlp;
      
      scheme_mlp->input_size=input_size;
      scheme[l].mlp.params_mlp->input_size = input_size;
      scheme[l].mlp.params_mlp->input_data=in->data;
      cnn_l->tag.mlp = create_nnet(scheme[l].mlp.scheme_mlp,cnn->tp,scheme[l].mlp.params_mlp);
      cnn->mlp_head=cnn_l->tag.mlp;
      cnn_l->tag.mlp->layers[0].grad_in = new_matrix_set_data( in->N,input_size,in->grad);
      cnn_l->tag.mlp->t_step = cnn->t_step;
      
    }
    
    }
  }
  return cnn;
}

void forward_cnnet(cnnet *cnn, STATE_RUN state_run) {
  for (uint32_t l = 0; l < cnn->n_layers; l++) {
    cnnet_layer *cnn_l=cnn->layers+l;
    tensor *in = cnn_l->in;
    tensor *out = cnn_l->out;
    uint32_t kh = cnn_l->kh;
    uint32_t kw = cnn_l->kw;
    uint32_t stride = cnn_l->stride;
    switch (cnn_l->l_type) {
    case CONV_LAYER: {
      conv_linear_l *conv=&cnn_l->tag.conv;
      matrix *in_mat  = conv->in_mat;
      matrix *W_mat   = conv->W_mat;
      matrix *out_mat = conv->out_mat;
      uint32_t padd   = conv->padding;
      im2col(in, in_mat, kw, kh, stride, padd);
      threaded_matmult(W_mat, in_mat, out_mat, NN, true, cnn->tp);
      matrix_sum_by_row(out_mat,conv->bias);
      matrix_to_tensor_NHWC(out_mat,out,1);
    }break;
    case ACTIV_CNN: {
      activ_func act = cnn_l->tag.activ.activation;
      for (int i = 0; i < out->len; i++) {
        float z = in->data[i];
        out->data[i] =  (act == RELU)    ? ReLU(z)
                      : (act == L_RELU)  ? leaky_ReLU(z)
                      : (act == SIGMOID) ? sigmoid(z)
                      : (act == TANH)    ? tanhf(z)
                                         : z;
      }
    }break;
    case POOLING_LAYER: {
      uint32_t *mask = cnn_l->tag.pool.mask;
      max_pooling(in, out, mask, kh, kw, stride, 0);
    }break;
    case MLP_LAYER: forward_pass(cnn_l->tag.mlp, state_run);
    break;
    }
  }
}

void backprop_cnnet(cnnet *cnn, matrix *labels) {

  for (uint32_t l = 0; l < cnn->n_layers; l++) {
    cnnet_layer *cnn_l=cnn->layers+l;
    if (cnn_l->in && cnn_l->in->grad)
      memset(cnn_l->in->grad, 0, cnn_l->in->len * sizeof(float));
    if (cnn_l->out && cnn_l->out->grad)
      memset(cnn_l->out->grad, 0, cnn_l->out->len * sizeof(float));
  }

  for (int32_t l = cnn->n_layers - 1; l >= 0; l--) {
    cnnet_layer *cnn_l=cnn->layers+l;
    tensor *in  = cnn_l->in;
    tensor *out = cnn_l->out;
    uint32_t kh = cnn_l->kh;
    uint32_t kw = cnn_l->kw;
    uint32_t stride = cnn_l->stride;
    switch (cnn_l->l_type) {
    case MLP_LAYER: backprop(labels, cnn_l->tag.mlp);
    break;
    case POOLING_LAYER: max_pooling_backward(out, in, cnn_l->tag.pool.mask);    
    break;
    case ACTIV_CNN: {
      for (int i = 0; i < in->len; i++) {
        activ_func act = cnn_l->tag.activ.activation;
        float dz = out->grad[i];
        float z = in->data[i];
        dz *=   (act == RELU)    ? d_ReLU(z)
              : (act == L_RELU)  ? d_leaky_ReLU(z)
              : (act == SIGMOID) ? d_sigmoid(z)
              : (act == SILU)    ? d_silu(z)
              : (act == TANH)    ? d_tanh(z)
                                 : z;
        in->grad[i] = dz;
      }
    }break;
    case CONV_LAYER: {
      conv_linear_l *conv=&cnn_l->tag.conv;
      matrix *dZ_mat    = conv->dZ_mat;
      matrix *W_mat     = conv->W_mat;
      matrix *dbias_mat = conv->d_bias;
      matrix *t_W_mat   = conv->t_W_mat;
      matrix *dW_mat    = conv->dW_mat;
      matrix *in_mat    = conv->in_mat;
      matrix *t_in_mat  = conv->t_in_mat;
      uint32_t padd     = conv->padding;
    
      tensor_to_matrix_NHWC(dZ_mat,out,0);
      memset(dbias_mat->data, 0, dbias_mat->len * sizeof(float));
      for (int k = 0; k < conv->filters; k++) {
        float sum = 0;
        for (int j = 0; j < dZ_mat->col; j++)
          sum += dZ_mat->data[k * dZ_mat->col + j];
        dbias_mat->data[k] += sum;
      }
      transpose_by(W_mat, t_W_mat);
      transpose_by(in_mat, t_in_mat);
      threaded_matmult(dZ_mat, t_in_mat, dW_mat, NN, true, cnn->tp);
      threaded_matmult(t_W_mat, dZ_mat, in_mat, NN, true, cnn->tp);
      
      col2im(in, in_mat, kh, kw, stride, padd);
      break;
    }
    }
  }
}

void update_cnnet(cnnet *cnn) {
  for (uint32_t l = 0; l < cnn->n_layers; l++) {
    cnnet_layer *cnn_l=cnn->layers+l;
    switch (cnn_l->l_type) {
    case CONV_LAYER: {

      matrix *W  = cnn_l->tag.conv.W_mat;
      matrix *mW = cnn_l->tag.conv.mW_mat;
      matrix *vW = cnn_l->tag.conv.vW_mat;
      matrix *dW = cnn_l->tag.conv.dW_mat;

      matrix *bias = cnn_l->tag.conv.bias;
      matrix *mb   = cnn_l->tag.conv.mb_mat;
      matrix *vb   = cnn_l->tag.conv.vb_mat;
      matrix *d_bias = cnn_l->tag.conv.d_bias;
      ADAMW_correction(W, mW, vW, dW, cnn->b1, cnn->b2, cnn->learn_rt,
                       cnn->t_step, 0);
      ADAMW_correction(bias, mb, vb, d_bias, cnn->b1, cnn->b2, cnn->learn_rt,
                       cnn->t_step, 0);
      break;
    }
    case MLP_LAYER: update_layers_adamw(cnn_l->tag.mlp);
    break;
    }
  }
  cnn->t_step++;
}

void run_cnnet(size_t epoch_max, cnnet *cnet, data_loader *dtl, STATE_RUN state,
               FILE *fout) {
  fputs("epoch,error,acc\n", fout);
  puts("\nstart\n");
  tensor *in = cnet->layers[0].in;
  matrix batch_in = {
      .row = in->N,
      .col = in->H * in->W * in->C,
      .data = in->data,
      .end = in->data_end,
      .len = in->N * in->H * in->W * in->C,
  };
  matrix *target = new_matrix(cnet->batch_size, cnet->n_labels);

  struct timespec t0, t1, delta;
  float error = 0, acc = 0, ba_md = (float)cnet->batch_size / dtl->size_lb;
  size_t end = dtl->n_itens / cnet->batch_size;
  matrix *out_layer=cnet->mlp_head->layers[cnet->mlp_head->n_layers - 1].out;
  for (size_t e = 0, i; e < epoch_max; e++) {
    error = 0;
    acc = 0;
    shuffle_data(dtl);
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (i = 0; i < end; i++) {
      load_batch(dtl, &batch_in, target, i);
      forward_cnnet(cnet, state);
      if (state == TRAIN) {
        backprop_cnnet(cnet, target);
        update_cnnet(cnet);
      }
      error += cat_cross_entropy(out_layer , target);
      acc += get_accuracy(out_layer, target) * 100.0;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    sub_timespec(t0, t1, &delta);
    printf("e = %2ld | dt(s) = %d.%09ld", e + 1, (int)delta.tv_sec,
           delta.tv_nsec);
    error *= ba_md;
    acc *= ba_md;
    printf(" loss = %.8f | acc = %.8f\n", error, acc);
    fprintf(fout, "%zu,%f,%f\n", e, error, acc);
  }
  puts("\nend.\n");

  destroy_matrix(target);
}

void train_cnnet(size_t epoch_max, cnnet *cnet, data_loader *dtl,
                 char *nmfile) {
  FILE *file_train = fopen(nmfile, "w");
  if (!file_train) {
    perror("\nERRO: nao foi possivel criar arquivo de treino.\n");
    exit(1);
  }
  run_cnnet(epoch_max, cnet, dtl, TRAIN, file_train);
  fclose(file_train);
}

void out_cnnet(cnnet *cnet, data_loader *dtl, char *namef) {
  FILE *file_test = fopen(namef, "w");
  if (!file_test) {
    perror("\nERRO: nao foi possivel criar arquivo de teste.\n");
    exit(1);
  }
  run_cnnet(1, cnet, dtl, TEST, file_test);
  fclose(file_test);
}

void im2col(tensor *input, matrix *buffer, int k_w, int k_h, int stride,
            int padding) {

  uint32_t ho = (input->H - k_h + 2 * padding) / stride + 1.f;
  uint32_t wo = (input->W - k_w + 2 * padding) / stride + 1.f;
  float *p_buff = buffer->data;
  for (int32_t r = 0; r < k_h; r++)
    for (int32_t s = 0; s < k_w; s++)
      for (int32_t c = 0; c < input->C; c++)
        for (int32_t b = 0; b < input->N; b++)
          for (int32_t m = 0; m < ho; m++) {
            int32_t in_h = m * stride - padding + r;
            for (int32_t n = 0; n < wo; n++, p_buff++) {
              int32_t in_w = n * stride - padding + s;
              if (in_h >= 0 && in_w >= 0 && in_h < input->H &&
                  in_w < input->W) {
                uint32_t pos =
                    ((b * input->H + in_h) * input->W + in_w) * input->C + c;
                *p_buff = input->data[pos];
              }
            }
          }
}

void col2im(tensor *input, matrix *buffer, int k_w, int k_h, int stride,
            int padding) {

  uint32_t ho = (input->H - k_h + 2 * padding) / stride + 1.f;
  uint32_t wo = (input->W - k_w + 2 * padding) / stride + 1.f;
  float *p_buff = buffer->data;
  for (int32_t r = 0; r < k_h; r++)
    for (int32_t s = 0; s < k_w; s++)
      for (int32_t c = 0; c < input->C; c++)
        for (int32_t b = 0; b < input->N; b++)
          for (int32_t m = 0; m < ho; m++) {
            int32_t in_h = m * stride - padding + r;
            for (int32_t n = 0; n < wo; n++, p_buff++) {
              int32_t in_w = n * stride - padding + s;
              if (in_h >= 0 && in_w >= 0 && in_h < input->H &&
                  in_w < input->W) {
                uint32_t pos =
                    ((b * input->H + in_h) * input->W + in_w) * input->C + c;
                input->grad[pos] += *p_buff;
              }
            }
          }
}

void max_pooling(tensor *in, tensor *out, uint32_t *mask, uint32_t kh,
                 uint32_t kw, uint32_t stride, uint32_t padd) {
  for (size_t n = 0; n < in->N; n++)
    for (size_t c = 0; c < in->C; c++)
      for (size_t h = 0; h < out->H; h++)
        for (size_t w = 0; w < out->W; w++) {
          float max = -INFINITY;
          uint32_t max_index = 0;
          for (size_t i = 0; i < kh; i++) {
            for (size_t j = 0; j < kw; j++) {
              int32_t p_h = h * stride + i - padd;
              int32_t p_w = w * stride + j - padd;
              if (p_h >= 0 && p_w >= 0 && p_h < in->H && p_w < in->W) {
                uint32_t pos = ((n * in->H + p_h) * in->W + p_w) * in->C + c;
                float val = in->data[pos];
                if (val > max) {
                  max = val;
                  max_index = pos;
                }
              }
            }
          }
          uint32_t out_idx = ((n * out->H + h) * out->W + w) * out->C + c;
          out->data[out_idx] = max;
          mask[out_idx] = max_index;
        }
}

void max_pooling_backward(tensor *dout, tensor *dinput, uint32_t *mask) {
  for (size_t i = 0; i < dout->len; i++) {
    dinput->grad[mask[i]] += dout->grad[i];
  }
}
