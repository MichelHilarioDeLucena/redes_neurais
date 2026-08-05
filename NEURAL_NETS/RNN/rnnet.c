#include "rnnet.h"
#include "../mmath_lib/mm_lib.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

rnnet *create_rnnet(scheme_rnn *scheme, uint32_t batch, uint32_t n_layers,
                    float lr, uint32_t t_step) {
  rnnet *rnn = calloc(1, sizeof(rnnet));

  rnn->learn_rt = lr;
  rnn->n_layers = n_layers;
  rnn->time_step = t_step;
  rnn->batch_size = batch;
  rnn->tp = new_thread_pool();
  rnn->layers = calloc(n_layers, sizeof(rnnet_layer));

  rnn->layers->in = calloc(t_step, sizeof(matrix *));
  for (uint32_t t = 0; t < t_step; t++)
    rnn->layers->in[t] = new_matrix(batch, scheme[0].input_size);

  for (uint32_t l = 0; l < n_layers; l++) {
    rnnet_layer *layer = rnn->layers + l;

    layer->activ = scheme[l].activ;

    layer->h = calloc(t_step + 1, sizeof(matrix *));
    layer->z = calloc(t_step, sizeof(matrix *));
    layer->dz = calloc(t_step, sizeof(matrix *));
    layer->dh = calloc(t_step, sizeof(matrix *));

    if (l > 0)
      layer->in = rnn->layers[l - 1].h + 1;
    for (uint32_t t = 0; t < t_step; t++) {
      layer->h[t] = new_matrix(batch, scheme[l].hidden_size);
      layer->z[t] = new_matrix(batch, scheme[l].hidden_size);
      layer->dz[t] = new_matrix(batch, scheme[l].hidden_size);
      layer->dh[t] = new_matrix(batch, scheme[l].hidden_size);
    }
    uint32_t wih_in = !l ? scheme[l].input_size : scheme[l - 1].hidden_size;
    layer->W_ih = new_matrix(wih_in, scheme[l].hidden_size);
    layer->dW_ih = new_matrix(wih_in, scheme[l].hidden_size);

    layer->W_hh = new_matrix(scheme[l].hidden_size, scheme[l].hidden_size);
    layer->dW_hh = new_matrix(scheme[l].hidden_size, scheme[l].hidden_size);

    if (layer->activ == RELU || layer->activ == L_RELU) {
      if (layer->activ == RELU) {
        init_uniform_distr_He(layer->W_ih, wih_in);

        memset(layer->W_hh->data, 0, layer->W_hh->len * sizeof(float));
        for (uint32_t i = 0; i < scheme[l].hidden_size; i++) {
          layer->W_hh->data[i * scheme[l].hidden_size + i] = 1.0f;
        }
      } else {

        init_uniform_distr_He(layer->W_ih, wih_in);
        init_uniform_distr_He(layer->W_hh, layer->W_hh->col);
      }
    } else {
      init_uniform_distr(layer->W_ih, wih_in, layer->W_ih->col);
      init_uniform_distr(layer->W_hh, layer->W_hh->col, layer->W_hh->col);
    }

    layer->b_h = new_matrix(1, scheme[l].hidden_size);
    layer->db_h = new_matrix(1, scheme[l].hidden_size);

    layer->y_h = new_matrix(batch, scheme[l].hidden_size);

    layer->h[t_step] = new_matrix(batch, scheme[l].hidden_size);
  }
  uint32_t last = n_layers - 1;
  rnnet_layer *last_l = rnn->layers + last;

  last_l->out = calloc(t_step, sizeof(matrix *));
  last_l->dout = calloc(t_step, sizeof(matrix *));
  for (uint32_t t = 0; t < t_step; t++)
    last_l->out[t] = new_matrix(batch, scheme[last].out_size),
    last_l->dout[t] = new_matrix(batch, scheme[last].out_size);

  last_l->W_oh = new_matrix(scheme[last].hidden_size, scheme[last].out_size);
  if (last_l->activ == RELU || last_l->activ == L_RELU)
    init_uniform_distr_He(last_l->W_oh, scheme[last].hidden_size);
  else
    init_uniform_distr(last_l->W_oh, scheme[last].hidden_size,
                       scheme[last].out_size);

  last_l->dW_oh = new_matrix(scheme[last].hidden_size, scheme[last].out_size);
  last_l->b_o = new_matrix(1, scheme[last].out_size);
  last_l->db_o = new_matrix(1, scheme[last].out_size);
  return rnn;
}

void forward_rnnet(rnnet *rnn) {
  for (uint32_t l = 0; l < rnn->n_layers; l++)
    memset(rnn->layers[l].h[0]->data, 0,
           rnn->layers[l].h[0]->len * sizeof(float));

  for (uint32_t t = 0; t < rnn->time_step; t++) {
    for (uint32_t l = 0; l < rnn->n_layers; l++) {
      rnnet_layer *layer = rnn->layers + l;

      threaded_matmult(layer->in[t], layer->W_ih, layer->z[t], NN, true,rnn->tp);
      threaded_matmult(layer->h[t], layer->W_hh, layer->y_h, NN, true,rnn->tp);
      matrix_sum(layer->z[t], layer->y_h, layer->z[t]);

      matrix_sum_broadcast(layer->z[t], layer->b_h);

      float *p_h = layer->h[t + 1]->data;
      float *end = layer->h[t + 1]->end;
      float *p_z = layer->z[t]->data;

      switch (layer->activ) {
      case RELU:
        while (p_h < end)
          *p_h++ = ReLU(*p_z++);
        break;
      case L_RELU:
        while (p_h < end)
          *p_h++ = leaky_ReLU(*p_z++);
        break;
      case SIGMOID:
        while (p_h < end)
          *p_h++ = sigmoid(*p_z++);
        break;
      case TANH:
        while (p_h < end)
          *p_h++ = tanhf(*p_z++);
        break;
      case SILU:
        while (p_h < end)
          *p_h++ = silu(*p_z++);
        break;
      }
      if (l == rnn->n_layers - 1 && t == rnn->time_step - 1) {
        threaded_matmult(layer->h[t + 1], layer->W_oh, layer->out[t], NN, true,rnn->tp);
        matrix_sum_broadcast(layer->out[t], layer->b_o);
        log_softmax(layer->out[t]);
      }
    }
  }
}

void backprop_rnnet(rnnet *rnn, matrix *target) {
  for (uint32_t l = 0; l < rnn->n_layers; l++) {
    rnnet_layer *layer = rnn->layers + l;
    for(uint32_t t=0;t<rnn->time_step;t++)
      memset(layer->dh[t]->data, 0, layer->dh[t]->len * sizeof(float));

    memset(layer->dW_ih->data, 0, layer->dW_ih->len * sizeof(float));
    memset(layer->dW_hh->data, 0, layer->dW_hh->len * sizeof(float));
    memset(layer->db_h->data, 0, layer->db_h->len * sizeof(float));
    if (l == rnn->n_layers - 1) {
      memset(layer->dW_oh->data, 0, layer->dW_oh->len * sizeof(float));
      memset(layer->db_o->data, 0, layer->db_o->len * sizeof(float));
    }
  }
  for (int32_t t = rnn->time_step - 1; t >= 0; t--) {
    for (int32_t l = rnn->n_layers - 1; l >= 0; l--) {
      rnnet_layer *layer = rnn->layers + l;
      if (l == rnn->n_layers - 1 && (t == rnn->time_step - 1)) {
        matrix_exp_sub(layer->out[t], target, layer->dout[t]);
        matrix_scalar_prod(layer->dout[t], 1.f / rnn->batch_size);
        threaded_matmult(layer->h[t + 1], layer->dout[t], layer->dW_oh, TN, false,rnn->tp);

        matrix *delta = layer->dout[t];
        matrix *b_bff = layer->db_o;
        for (uint32_t r = 0; r < delta->row; r++)
          for (uint32_t c = 0; c < delta->col; c++)
            b_bff->data[c] += delta->data[r * delta->col + c];

        threaded_matmult(layer->dout[t], layer->W_oh, layer->dh[t], NT, true,rnn->tp);
      } else if (l == rnn->n_layers - 1 && t < rnn->time_step - 1) {
        threaded_matmult(layer->dz[t + 1], layer->W_hh, layer->dh[t], NT, true,rnn->tp);
      } else {
        threaded_matmult((layer + 1)->dz[t], (layer + 1)->W_ih, layer->dh[t], NT,
                    true,rnn->tp);
        if (t < rnn->time_step - 1)
          threaded_matmult(layer->dz[t + 1], layer->W_hh, layer->dh[t], NT, false,rnn->tp);
      }

      float *p_z = layer->z[t]->data;
      float *p_dz = layer->dz[t]->data;
      float *p_dh = layer->dh[t]->data;
      float *end = layer->dz[t]->end;
      switch (layer->activ) {
      case RELU:
        while (p_dz < end)
          *p_dz++ = *p_dh++ * d_ReLU(*p_z++);
        break;
      case L_RELU:
        while (p_dz < end)
          *p_dz++ = *p_dh++ * d_leaky_ReLU(*p_z++);
        break;
      case SIGMOID:
        while (p_dz < end)
          *p_dz++ = *p_dh++ * d_sigmoid(*p_z++);
        break;
      case TANH:
        while (p_dz < end)
          *p_dz++ = *p_dh++ * d_tanh(*p_z++);
        break;
      case SILU:
        while (p_dz < end)
          *p_dz++ = *p_dh++ * d_silu(*p_z++);
        break;
      }
      threaded_matmult(layer->h[t], layer->dz[t], layer->dW_hh, TN, false,rnn->tp);
      threaded_matmult(layer->in[t], layer->dz[t], layer->dW_ih, TN, false,rnn->tp);
      matrix *delta = layer->dz[t];
      matrix *b_bff = layer->db_h;
      for (uint32_t r = 0; r < delta->row; r++)
        for (uint32_t c = 0; c < delta->col; c++)
          b_bff->data[c] += delta->data[r * delta->col + c];
    }
  }
}

void update_rnnet(rnnet *rnn) {
  float max_norm = 5.0;
  for (uint32_t l = 0; l < rnn->n_layers; l++) {
    rnnet_layer *layer = rnn->layers + l;
    SGD(layer->W_ih, layer->dW_ih, rnn->learn_rt, max_norm);
    SGD(layer->W_hh, layer->dW_hh, rnn->learn_rt, max_norm);
    SGD(layer->b_h, layer->db_h, rnn->learn_rt, max_norm);
    if (l == rnn->n_layers - 1)
      SGD(layer->W_oh, layer->dW_oh, rnn->learn_rt, max_norm),
          SGD(layer->b_o, layer->db_o, rnn->learn_rt, max_norm);
  }
}

void run_nnet(size_t epoch_max, rnnet *rnn, data_loader *dtl, STATE_RUN state,
              FILE *fout) {
  fputs("epoch,loss,acc\n", fout);
  puts("\nstart\n");

  matrix *out = rnn->layers[rnn->n_layers - 1].out[rnn->time_step - 1];
  matrix *target = new_matrix(rnn->batch_size, out->col);

  struct timespec t0, t1, delta;
  float loss = 0, acc = 0, ba_md = (float)rnn->batch_size / dtl->size_lb;
  float lr_og = rnn->learn_rt;
  size_t end = dtl->n_itens / rnn->batch_size;
  for (size_t e = 0, i; e < epoch_max; e++) {
    loss = 0;
    acc = 0;

    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (i = 0; i < end; i++) {
      for (uint32_t t = 0; t < rnn->time_step; t++)
        load_batch_input(dtl, rnn->layers[0].in[t], rnn->time_step * t, i);

      load_batch_label(dtl, target, i);

      forward_rnnet(rnn);
      if (state == TRAIN) {
        backprop_rnnet(rnn, target);
        update_rnnet(rnn);
      }
      loss += cat_cross_entropy(out, target);
      acc += get_accuracy(out, target) * 100.0;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    sub_timespec(t0, t1, &delta);
    printf("e = %ld | dt(s) = %d.%.4ld", e + 1, (int)delta.tv_sec,
           delta.tv_nsec);

    loss *= ba_md;
    acc *= ba_md;
    printf(" loss = %.4f | acc = %.4f\n", loss, acc);
    fprintf(fout, "%zu,%f,%f\n", e, loss, acc);
    if (state == TRAIN)
      shuffle_data(dtl);
  }
  puts("\nend.\n");
  destroy_matrix(target);
}

void train_nnet(size_t epoch_max, rnnet *rnn, data_loader *dtl, char *nmfile) {
  FILE *file_train = fopen(nmfile, "w");
  if (!file_train) {
    perror("\nERRO: nao foi possivel criar arquivo de treino.\n");
    exit(1);
  }
  run_nnet(epoch_max, rnn, dtl, TRAIN, file_train);
  fclose(file_train);
}

void out_nnet(rnnet *rnn, data_loader *dtl, char *namef) {
  FILE *file_test = fopen(namef, "w");
  if (!file_test) {
    perror("\nERRO: nao foi possivel criar arquivo de teste.\n");
    exit(1);
  }
  run_nnet(1, rnn, dtl, TEST, file_test);
  fclose(file_test);
}
