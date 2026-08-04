#include "cnnet.h"
#include <math.h>
#include <stdlib.h>

cnnet *create_cnnet(scheme_cnn *scheme, uint32_t n_layers,
                    cnnet_params *params) {
  cnnet *cnn = malloc(sizeof(cnnet));
  cnn->batch_size = params->in_N;
  cnn->n_layers = n_layers;
  cnn->layers = malloc(sizeof(cnnet_layer) * n_layers);
  cnn->learn_rt = params->learn_rt;
  cnn->n_labels = params->n_labels;
  cnn->t_step = 1;
  cnn->b1 = .9f;
  cnn->b2 = .999f;
  cnn->tp = new_thread_pool();

  cnn->layers[0].input = new_tensor_grad_init(params->in_N, params->in_H,
                                              params->in_W, params->in_C);

  for (uint32_t l = 0; l < n_layers; l++) {
    cnn->layers[l].l_type = scheme[l].type;
    cnn->layers[l].activation = scheme[l].activation;
    cnn->layers[l].stride = scheme[l].stride;
    cnn->layers[l].kh = scheme[l].kh;
    cnn->layers[l].kw = scheme[l].kw;

    if (l > 0)
      cnn->layers[l].input = cnn->layers[l - 1].output;

    tensor *in = cnn->layers[l].input;

    switch (cnn->layers[l].l_type) {
    case CONV_LAYER: {
      uint32_t kh = scheme[l].kh;
      uint32_t kw = scheme[l].kw;
      uint32_t ho =
          (in->H - kh + 2 * scheme[l].conv.padding) / scheme[l].stride + 1;
      uint32_t wo =
          (in->W - kw + 2 * scheme[l].conv.padding) / scheme[l].stride + 1;
      uint32_t filters = scheme[l].conv.filters;
      uint32_t M = in->C * kw * kh;
      uint32_t N = ho * wo * params->in_N;

      cnn->layers[l].z_out = new_tensor_grad_init(params->in_N, ho, wo, filters);
      cnn->layers[l].output = new_tensor_grad_init(params->in_N, ho, wo, filters);
      cnn->layers[l].l_tag.conv.filters = filters;
      cnn->layers[l].l_tag.conv.padding = scheme[l].conv.padding;

      cnn->layers[l].l_tag.conv.in_mat = new_matrix(M, N);
      cnn->layers[l].l_tag.conv.W_mat = new_matrix(filters, M);
      cnn->layers[l].l_tag.conv.mW_mat = new_matrix(filters, M);
      cnn->layers[l].l_tag.conv.vW_mat = new_matrix(filters, M);

      cnn->layers[l].l_tag.conv.bias = new_matrix(filters, 1);

      cnn->layers[l].l_tag.conv.mb_mat = new_matrix(filters, 1);
      cnn->layers[l].l_tag.conv.vb_mat = new_matrix(filters, 1);
      cnn->layers[l].l_tag.conv.dW_mat = new_matrix(filters, M);
      cnn->layers[l].l_tag.conv.d_bias = new_matrix(filters, 1);
      cnn->layers[l].l_tag.conv.dZ_mat = new_matrix(filters, N);
      cnn->layers[l].l_tag.conv.out_mat = new_matrix(filters, N);

      cnn->layers[l].l_tag.conv.t_W_mat = new_matrix(M, filters);
      cnn->layers[l].l_tag.conv.t_in_mat = new_matrix(N, M);
      if (params->rstate) {
        if (cnn->layers[l].activation == RELU ||
            cnn->layers[l].activation == L_RELU)
          init_uniform_distr_He_xors64(cnn->layers[l].l_tag.conv.W_mat, M,
                                       params->rstate);
        else
          init_uniform_distr_xors64(cnn->layers[l].l_tag.conv.W_mat, M, filters,
                                    params->rstate);
      } else {
        if (cnn->layers[l].activation == RELU ||
            cnn->layers[l].activation == L_RELU)
          init_uniform_distr_He(cnn->layers[l].l_tag.conv.W_mat, M);
        else
          init_uniform_distr(cnn->layers[l].l_tag.conv.W_mat, M, filters);
      }
      break;
    }
    case POOLING_LAYER: {
      uint32_t kh = scheme[l].kh, kw = scheme[l].kw;
      uint32_t ho = (in->H - kh) / scheme[l].stride + 1;
      uint32_t wo = (in->W - kw) / scheme[l].stride + 1;
      cnn->layers[l].z_out = NULL;
      cnn->layers[l].output = new_tensor_grad_init(params->in_N, ho, wo, in->C);
      cnn->layers[l].l_tag.pool.mask =
          calloc(cnn->layers[l].output->len, sizeof(uint32_t));
      break;
    }
    case DENSE_LAYER: {
      uint32_t input_size = in->H * in->W * in->C;
      scheme[l].dense.layers[0] = input_size;

      cnn->mlp_head = create_nnet(
          scheme[l].dense.layers, scheme[l].dense.n_layers, params->in_N,
          input_size, scheme[l].activation, cnn->tp, params->learn_rt,
          params->w_dec, params->pval, params->rstate);
      cnn->layers[l].l_tag.dense.mlp = cnn->mlp_head;

      cnn->mlp_head->input_grad = malloc(sizeof(matrix));
      *cnn->mlp_head->input_grad = (matrix){.row = in->N,
                                            .col = input_size,
                                            .data = in->grad,
                                            .end = in->grad_end,
                                            .len = in->len};
      cnn->mlp_head->t_step = cnn->t_step;
      break;
    }
    }
  }
  return cnn;
}

void forward_cnnet(cnnet *cnn, STATE_RUN state_run) {
  for (uint32_t l = 0; l < cnn->n_layers; l++) {

    tensor *in = cnn->layers[l].input;
    tensor *z_out = cnn->layers[l].z_out;
    tensor *output = cnn->layers[l].output;
    uint32_t kh = cnn->layers[l].kh;
    uint32_t kw = cnn->layers[l].kw;
    uint32_t stride = cnn->layers[l].stride;
    switch (cnn->layers[l].l_type) {
    case CONV_LAYER: {
      matrix *in_mat = cnn->layers[l].l_tag.conv.in_mat;
      matrix *W_mat = cnn->layers[l].l_tag.conv.W_mat;
      matrix *out_mat = cnn->layers[l].l_tag.conv.out_mat;
      uint32_t padd = cnn->layers[l].l_tag.conv.padding;

      im2col(in, in_mat, kw, kh, stride, padd);

      threaded_matmult(W_mat, in_mat, out_mat, NN, true, cnn->tp);
      for (int f = 0; f < out_mat->row; f++) {
        float b_val = cnn->layers[l].l_tag.conv.bias->data[f];
        for (int i = 0; i < out_mat->col; i++)
          out_mat->data[f * out_mat->col + i] += b_val;
      }
      for (int b = 0; b < in->N; b++)
        for (int h = 0; h < z_out->H; h++)
          for (int w = 0; w < z_out->W; w++) {
            int patch = b * (z_out->H * z_out->W) + h * z_out->W + w;
            for (int k = 0; k < z_out->C; k++) {
              float val = out_mat->data[k * out_mat->col + patch];
              int out_idx = ((b * z_out->H + h) * z_out->W + w) * z_out->C + k;
              z_out->data[out_idx] = val;
            }
          }

      AFUNC_TYPE act = cnn->layers[l].activation;
      for (int i = 0; i < z_out->len; i++) {
        float z = z_out->data[i];
        output->data[i] = (act == RELU)      ? ReLU(z)
                          : (act == L_RELU)  ? leaky_ReLU(z)
                          : (act == SIGMOID) ? sigmoid(z)
                          : (act == TANH)    ? tanhf(z)
                                             : z;
      }
      break;
    }
    case POOLING_LAYER: {
      uint32_t *mask = cnn->layers[l].l_tag.pool.mask;
      max_pooling(in, output, mask, kh, kw, stride, 0);
      break;
    }
    case DENSE_LAYER: {
      uint32_t features = in->H * in->W * in->C;
      matrix dense_in = {
          .row = in->N,
          .col = features,
          .data = in->data,
          .end = in->data_end,
          .len = in->len,
      };
      forward_pass(&dense_in, cnn->layers[l].l_tag.dense.mlp, state_run);
      break;
    }
    }
  }
}

void backprop_cnnet(cnnet *cnn, matrix *labels) {

  for (uint32_t l = 0; l < cnn->n_layers; l++) {
    if (cnn->layers[l].input && cnn->layers[l].input->grad)
      memset(cnn->layers[l].input->grad, 0,
             cnn->layers[l].input->len * sizeof(float));
    if (cnn->layers[l].z_out && cnn->layers[l].z_out->grad)
      memset(cnn->layers[l].z_out->grad, 0,
             cnn->layers[l].z_out->len * sizeof(float));
    if (cnn->layers[l].output && cnn->layers[l].output->grad)
      memset(cnn->layers[l].output->grad, 0,
             cnn->layers[l].output->len * sizeof(float));
  }

  for (int32_t l = cnn->n_layers - 1; l >= 0; l--) {

    tensor *in = cnn->layers[l].input;
    tensor *z_out = cnn->layers[l].z_out;
    tensor *output = cnn->layers[l].output;
    uint32_t kh = cnn->layers[l].kh;
    uint32_t kw = cnn->layers[l].kw;
    uint32_t stride = cnn->layers[l].stride;
    switch (cnn->layers[l].l_type) {
    case DENSE_LAYER: {
      nnet *mlp = cnn->layers[l].l_tag.dense.mlp;
      backprop(labels, cnn->layers[l].l_tag.dense.mlp);
      break;
    }
    case POOLING_LAYER: {
      max_pooling_backward(output, in, cnn->layers[l].l_tag.pool.mask);
      break;
    }
    case CONV_LAYER: {
      matrix *dZ_mat = cnn->layers[l].l_tag.conv.dZ_mat;
      matrix *W_mat = cnn->layers[l].l_tag.conv.W_mat;
      matrix *dbias_mat = cnn->layers[l].l_tag.conv.d_bias;
      matrix *t_W_mat = cnn->layers[l].l_tag.conv.t_W_mat;
      matrix *dW_mat = cnn->layers[l].l_tag.conv.dW_mat;
      matrix *in_mat = cnn->layers[l].l_tag.conv.in_mat;
      matrix *t_in_mat = cnn->layers[l].l_tag.conv.t_in_mat;
      uint32_t padd = cnn->layers[l].l_tag.conv.padding;

      
      memset(dbias_mat->data, 0, dbias_mat->len * sizeof(float));

      transpose_by(W_mat, t_W_mat);
      transpose_by(in_mat, t_in_mat);

      for (int i = 0; i < z_out->len; i++) {
        float dz = output->grad[i];
        float z = z_out->data[i];
        AFUNC_TYPE act = cnn->layers[l].activation;
        dz *= (act == RELU)      ? d_ReLU(z)
              : (act == L_RELU)  ? d_leaky_ReLU(z)
              : (act == SIGMOID) ? d_sigmoid(z)
              : (act == TANH)    ? d_tanh(z)
                                 : z;
        z_out->grad[i] = dz;
      }
      for (int b = 0; b < output->N; b++)
        for (int h = 0; h < output->H; h++)
          for (int w = 0; w < output->W; w++) {
            int patch = b * (output->H * output->W) + h * output->W + w;
            for (int k = 0; k < output->C; k++) {
              int idx = ((b * output->H + h) * output->W + w) * output->C + k;
              dZ_mat->data[k * dZ_mat->col + patch] = z_out->grad[idx];
            }
          }

      for (int k = 0; k < cnn->layers[l].l_tag.conv.filters; k++) {
        float sum = 0;
        for (int j = 0; j < dZ_mat->col; j++)
          sum += dZ_mat->data[k * dZ_mat->col + j];
        dbias_mat->data[k] += sum;
      }
      threaded_matmult(dZ_mat, t_in_mat, dW_mat, NN, true, cnn->tp);
      threaded_matmult(t_W_mat, dZ_mat, in_mat, NN, true, cnn->tp);
      memset(in_mat->data, 0, in_mat->len * sizeof(float));
      col2im(in, in_mat, kh, kw, stride, padd);
      break;
    }
    }
  }
}

void update_cnnet(cnnet *cnn) {
  for (uint32_t l = 0; l < cnn->n_layers; l++) {
    switch (cnn->layers[l].l_type) {
    case CONV_LAYER: {

      matrix *W = cnn->layers[l].l_tag.conv.W_mat;

      matrix *mW = cnn->layers[l].l_tag.conv.mW_mat;
      matrix *vW = cnn->layers[l].l_tag.conv.vW_mat;
      matrix *dW = cnn->layers[l].l_tag.conv.dW_mat;

      matrix *bias = cnn->layers[l].l_tag.conv.bias;
      matrix *mb = cnn->layers[l].l_tag.conv.mb_mat;
      matrix *vb = cnn->layers[l].l_tag.conv.vb_mat;
      matrix *d_bias = cnn->layers[l].l_tag.conv.d_bias;
      ADAMW_correction(W, mW, vW, dW, cnn->b1, cnn->b2, cnn->learn_rt,
                       cnn->t_step, 0);
      ADAMW_correction(bias, mb, vb, d_bias, cnn->b1, cnn->b2, cnn->learn_rt,
                       cnn->t_step, 0);
      break;
    }
    case DENSE_LAYER: {
      update_layers_adamw(cnn->layers[l].l_tag.dense.mlp);
      break;
    }
    }
  }
  cnn->t_step++;
}

void run_cnnet(size_t epoch_max, cnnet *cnet, data_loader *dtl, STATE_RUN state,
               FILE *fout) {
  fputs("epoch,error,acc\n", fout);
  puts("\nstart\n");
  tensor *in = cnet->layers[0].input;
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
      error += cat_cross_entropy(
          cnet->mlp_head->outputs[cnet->mlp_head->n_layers - 1], target);
      acc += get_accuracy(cnet->mlp_head->outputs[cnet->mlp_head->n_layers - 1],
                          target) *
             100.0;
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
                input->data[pos] += *p_buff;
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
