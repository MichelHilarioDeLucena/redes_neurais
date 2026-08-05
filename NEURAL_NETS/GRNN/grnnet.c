#include "grnnet.h"
#include "../mmath_lib/mm_lib.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

grnnet *create_grnnet(scheme_grnn *scheme, params_grrn *grn_params) {
  grnnet *grnn = calloc(1, sizeof(grnnet));
  float lr = grn_params->lr;
  uint32_t n_layers = grn_params->n_layers;
  uint32_t t_step = grn_params->t_step;
  uint32_t batch = grn_params->batch;
  grnn_mode mode = grn_params->mode;

  grnn->learn_rt = lr;
  grnn->n_layers = n_layers;
  grnn->time_step = t_step;
  grnn->batch_size = batch;
  grnn->mode = mode;
  grnn->tp = new_thread_pool();
  grnn->layers = calloc(n_layers, sizeof(grnnet_layer));

  grnn->layers->in = calloc(t_step, sizeof(matrix *));
  for (uint32_t t = 0; t < t_step; t++)
    grnn->layers->in[t] = new_matrix(batch, scheme[0].input_size);

  if (grn_params->mode & GRAD_IN) {
    grnn->layers->grad_in = calloc(t_step, sizeof(matrix *));
    for (uint32_t t = 0; t < t_step; t++)
      grnn->layers->grad_in[t] = new_matrix(batch, scheme[0].input_size);
    }
  

  for (uint32_t l = 0; l < n_layers; l++) {
    grnnet_layer *grn_l = grnn->layers + l;
    grn_l->t_layer = scheme[l].type;
    if (l > 0) {
      grn_l->in = (grn_l - 1)->out;
      grnn->layers->t_i = new_matrix(grn_l->in[0]->col,grn_l->in[0]->row);
      grn_l->grad_in = (grn_l - 1)->grad_out;
    }

    switch (grn_l->t_layer) {
    case GRU: {
      uint32_t h_size = scheme[l].config.gru.hidden_size;
      struct gru_layer *gru = &grn_l->layer.gru;
      grn_l->y1 = new_matrix(batch, h_size);
      grn_l->y2 = new_matrix(batch, h_size);
      grn_l->y3 = new_matrix(batch, h_size);
      grn_l->y4 = new_matrix(batch, h_size);

      grn_l->t_g = new_matrix(h_size, batch);
      grn_l->t_wh = new_matrix(h_size, h_size);
      
      gru->h = calloc(t_step, sizeof(matrix *));
      gru->dh = calloc(t_step, sizeof(matrix *));
      gru->z = calloc(t_step, sizeof(matrix *));
      gru->dz = calloc(t_step, sizeof(matrix *));
      gru->r = calloc(t_step, sizeof(matrix *));
      gru->dr = calloc(t_step, sizeof(matrix *));
      gru->n = calloc(t_step, sizeof(matrix *));
      gru->dn = calloc(t_step, sizeof(matrix *));

      for (uint32_t t = 0; t < t_step; t++) {
        gru->h[t] = new_matrix(batch, h_size);
        gru->dh[t] = new_matrix(batch, h_size);
        gru->z[t] = new_matrix(batch, h_size);
        gru->dz[t] = new_matrix(batch, h_size);
        gru->r[t] = new_matrix(batch, h_size);
        gru->dr[t] = new_matrix(batch, h_size);
        gru->n[t] = new_matrix(batch, h_size);
        gru->dn[t] = new_matrix(batch, h_size);
      }

      uint32_t wih_in = scheme[0].input_size;
      if (l > 0) {
        grnn_layer_t prev_type = scheme[l - 1].type;
        wih_in = scheme[l].type == GRU
                     ? scheme[l - 1].config.gru.hidden_size
                     : scheme[l - 1]
                           .config.dense.output_size;
      }
      grn_l->t_wi = new_matrix(h_size, wih_in);

      gru->W_iz = new_matrix(wih_in, h_size);
      gru->dW_iz = new_matrix(wih_in, h_size);
      gru->W_ir = new_matrix(wih_in, h_size);
      gru->dW_ir = new_matrix(wih_in, h_size);
      gru->W_in = new_matrix(wih_in, h_size);
      gru->dW_in = new_matrix(wih_in, h_size);

      gru->W_hz = new_matrix(h_size, h_size);
      gru->dW_hz = new_matrix(h_size, h_size);
      gru->W_hr = new_matrix(h_size, h_size);
      gru->dW_hr = new_matrix(h_size, h_size);
      gru->W_hn = new_matrix(h_size, h_size);
      gru->dW_hn = new_matrix(h_size, h_size);

      gru->tW_iz = new_matrix(h_size, wih_in);
      gru->tW_ir = new_matrix(h_size, wih_in);
      gru->tW_in = new_matrix(h_size, wih_in);
      gru->tW_hz = new_matrix(h_size, h_size);
      gru->tW_hr = new_matrix(h_size, h_size);
      gru->tW_hn = new_matrix(h_size, h_size);
      

      init_uniform_distr(gru->W_iz, wih_in, h_size);
      init_uniform_distr(gru->W_ir, wih_in, h_size);
      init_uniform_distr(gru->W_in, wih_in, h_size);
      init_uniform_distr(gru->W_hz, h_size, h_size);
      init_uniform_distr(gru->W_hr, h_size, h_size);
      init_uniform_distr(gru->W_hn, h_size, h_size);

      gru->b_z = new_matrix(1, h_size);
      gru->db_z = new_matrix(1, h_size);
      gru->b_r = new_matrix(1, h_size);
      gru->db_r = new_matrix(1, h_size);
      gru->b_n = new_matrix(1, h_size);
      gru->db_n = new_matrix(1, h_size);

      grn_l->out = gru->h;
      grn_l->grad_out = gru->dh;

    } break;
    case BATCHNORM: {

    } break;
    case DENSE: {
      struct dense_layer *dense = &grn_l->layer.dense;
      grnn_layer_t prev_type = scheme[l - 1].type;
      uint32_t out_size = scheme[l].config.dense.output_size;
      uint32_t in_size = in_size =
          (prev_type == GRU)     ? scheme[l - 1].config.gru.hidden_size
          : (prev_type == DENSE) ? scheme[l - 1].config.dense.output_size
                                 : 0;
      grn_l->out = calloc(t_step, sizeof(matrix *));
      grn_l->grad_out = calloc(t_step, sizeof(matrix *));

      for (uint32_t t = 0; t < t_step; t++)
        grn_l->out[t] = new_matrix(batch, out_size),
        grn_l->grad_out[t] = new_matrix(batch, out_size);

      dense->W = new_matrix(in_size, out_size);
      init_uniform_distr(dense->W, in_size, out_size);
      dense->dW = new_matrix(in_size, out_size);
      dense->b = new_matrix(1, out_size);
      dense->db = new_matrix(1, out_size);
    } break;
    }
  }
  if(!grnn){
    perror("REDE SEM MODO!.\n");
  }
  return grnn;
}

void forward_grnnet(grnnet *grnn) {

  for (uint32_t t = 0; t < grnn->time_step; t++) {
    for (uint32_t l = 0; l < grnn->n_layers; l++) {
      grnnet_layer *grn_l = grnn->layers + l;
      switch (grn_l->t_layer) {
      case GRU: {
        struct gru_layer *gru = &grn_l->layer.gru;
        threaded_matmult(grn_l->in[t], gru->W_iz, gru->z[t], NN, true,
                         grnn->tp);
        threaded_matmult(grn_l->in[t], gru->W_ir, gru->r[t], NN, true,
                         grnn->tp);
        if (t > 0) {
          threaded_matmult(gru->h[t - 1], gru->W_hz, grn_l->y1, NN, true,
                           grnn->tp);
          threaded_matmult(gru->h[t - 1], gru->W_hr, grn_l->y2, NN, true,
                           grnn->tp);
          matrix_sum(gru->z[t], grn_l->y1, gru->z[t]);
          matrix_sum(gru->r[t], grn_l->y2, gru->r[t]);
        }

        matrix_sum_broadcast(gru->z[t], gru->b_z);
        matrix_sum_broadcast(gru->r[t], gru->b_r);

        float *end = gru->z[t]->end;
        float *p_z = gru->z[t]->data, *p_r = gru->r[t]->data;
        float *p_n = gru->n[t]->data, *p_h = gru->h[t]->data;
        for (; p_z < end; p_z++, p_r++)
          *p_z = sigmoid(*p_z), *p_r = sigmoid(*p_r);

        if (t > 0) {

          matrix_hadd_dot(gru->r[t], gru->h[t - 1], grn_l->y1);
          threaded_matmult(grn_l->y1, gru->W_hn, grn_l->y2, NN, true, grnn->tp);
          threaded_matmult(grn_l->in[t], gru->W_in, grn_l->y1, NN, true,
                           grnn->tp);
          matrix_sum(grn_l->y1, grn_l->y2, gru->n[t]);
        } else {
          threaded_matmult(grn_l->in[t], gru->W_in, gru->n[t], NN, true,
                           grnn->tp);
        }

        matrix_sum_broadcast(gru->n[t], gru->b_n);

        end = gru->n[t]->end;
        for (end = gru->n[t]->end; p_n < end; p_n++)
          *p_n = tanhf(*p_n);
        matrix_scalar_k_sub_b(grn_l->y1, 1.f, gru->z[t]);
        if (t > 0) {
          matrix_hadd_dot(grn_l->y1, gru->n[t], grn_l->y2);
          matrix_hadd_dot(gru->z[t], gru->h[t - 1], grn_l->y1);
          matrix_sum(grn_l->y1, grn_l->y2, gru->h[t]);
        } else {
          matrix_hadd_dot(grn_l->y1, gru->n[t], gru->h[t]);
        }

      } break;

      case DENSE: {
        struct dense_layer *dense = &grn_l->layer.dense;
        if (grnn->mode == MANY_TO_ONE && t == grnn->time_step - 1) {
          threaded_matmult(grn_l->in[t], dense->W, grn_l->out[0], NN, true,
                           grnn->tp);

          matrix_sum_broadcast(grn_l->out[0], dense->b);
          log_softmax(grn_l->out[0]);
        }
        if (grnn->mode == MANY_TO_MANY) {
          threaded_matmult(grn_l->in[t], dense->W, grn_l->out[t], NN, true,
                           grnn->tp);
          matrix_sum_broadcast(grn_l->out[t], dense->b);
          log_softmax(grn_l->out[t]);
        }
      } break;
      }
    }
  }
}

void backprop_grnnet(grnnet *grnn, matrix *target) {
  for (uint32_t l = 0; l < grnn->n_layers; l++) {
    grnnet_layer *grn_l = grnn->layers + l;
    switch (grn_l->t_layer) {
    case GRU: {
      struct gru_layer *gru = &grn_l->layer.gru;
      for (uint32_t t = 0; t < grnn->time_step; t++)
        memset(gru->dh[t]->data, 0, gru->dh[t]->len * sizeof(float));
      memset(gru->dW_iz->data, 0, gru->dW_iz->len * sizeof(float));
      memset(gru->dW_ir->data, 0, gru->dW_ir->len * sizeof(float));
      memset(gru->dW_in->data, 0, gru->dW_in->len * sizeof(float));
      memset(gru->dW_hz->data, 0, gru->dW_hz->len * sizeof(float));
      memset(gru->dW_hr->data, 0, gru->dW_hr->len * sizeof(float));
      memset(gru->dW_hn->data, 0, gru->dW_hn->len * sizeof(float));
      memset(gru->db_z->data, 0, gru->db_z->len * sizeof(float));
      memset(gru->db_r->data, 0, gru->db_r->len * sizeof(float));
      memset(gru->db_n->data, 0, gru->db_n->len * sizeof(float));
    } break;
    case DENSE: {
      struct dense_layer *dense = &grn_l->layer.dense;
      memset(dense->dW->data, 0, dense->dW->len * sizeof(float));
      memset(dense->db->data, 0, dense->db->len * sizeof(float));
    } break;
    }
  }
  // struct timespec t0, t1, delta_t;
  // clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int32_t t = grnn->time_step - 1; t >= 0; t--) {
    for (int32_t l = grnn->n_layers - 1; l >= 0; l--) {
      grnnet_layer *grn_l = grnn->layers + l;

      switch (grn_l->t_layer) {
      case DENSE: {
        struct dense_layer *dense = &grn_l->layer.dense;
        

        if (grnn->mode == MANY_TO_ONE && t == grnn->time_step - 1) {

          matrix_exp_sub(grn_l->out[0], target, grn_l->grad_out[0]);
          matrix_scalar_prod(grn_l->grad_out[0], 1.f / grnn->batch_size);
          threaded_matmult(grn_l->in[t], grn_l->grad_out[0], dense->dW, TN,
                           false, grnn->tp);

          matrix *delta = grn_l->grad_out[0];
          matrix *b_bff = dense->db;
          for (uint32_t r = 0; r < delta->row; r++)
            for (uint32_t c = 0; c < delta->col; c++)
              b_bff->data[c] += delta->data[r * delta->col + c];
          threaded_matmult(grn_l->grad_out[0], dense->W,
                           grnn->layers[l - 1].grad_out[t], NT, true, grnn->tp);
        }

        if (grnn->mode == MANY_TO_MANY) {
          matrix_exp_sub(grn_l->out[t], target, grn_l->grad_out[t]);
          matrix_scalar_prod(grn_l->grad_out[t], 1.f / grnn->batch_size);
          threaded_matmult(grn_l->in[t], grn_l->grad_out[t], dense->dW, TN,
                           false, grnn->tp);

          matrix *delta = grn_l->grad_out[t];
          matrix *b_bff = dense->db;
          for (uint32_t r = 0; r < delta->row; r++)
            for (uint32_t c = 0; c < delta->col; c++)
              b_bff->data[c] += delta->data[r * delta->col + c];
          threaded_matmult(grn_l->grad_out[t], dense->W,
                           grnn->layers[l - 1].grad_out[t], NT, true, grnn->tp);
        }

      } break;
      case GRU: {
        
        struct gru_layer *gru = &grn_l->layer.gru;

        float *p_z = gru->z[t]->data, *end = gru->z[t]->end;
        float *p_n = gru->n[t]->data;
        float *p_dh = gru->dh[t]->data;
        float *p_dz = gru->dz[t]->data;
        float *p_dn = gru->dn[t]->data;

        if (t > 0) {
          float *p_h = gru->h[t - 1]->data;
          for (; p_z < end; p_z++, p_n++, p_h++, p_dh++, p_dz++, p_dn++) {
            float dh_next = *p_dh, z_val = *p_z, n_val = *p_n;
            *p_dz = dh_next * (*p_h - *p_n) * z_val * (1.f - *p_z);
            *p_dn = dh_next * (1.f - *p_z) * (1.f - n_val * n_val);
          }
        } else {
          for (; p_z < end; p_z++, p_n++, p_dh++, p_dz++, p_dn++) {
            float dh_next = *p_dh, z_val = *p_z, n_val = *p_n;
            *p_dz = dh_next * (-*p_n) * z_val * (1.f - *p_z);
            *p_dn = dh_next * (1.f - *p_z) * (1.f - n_val * n_val);
          }
        }
        
        threaded_matmult(gru->dz[t], gru->tW_hz, grn_l->y2, NN, true, grnn->tp);
        threaded_matmult(gru->dn[t], gru->tW_hn, grn_l->y1, NN, true, grnn->tp);

        float *p_o1 = grn_l->y1->data;
        float *p_o2 = grn_l->y2->data;
        float *p_o4 = grn_l->y4->data, *p_o3;

        float *p_r = gru->r[t]->data;
        if (t > 0) {
          float *p_dr = gru->dr[t]->data;
          float *p_h = gru->h[t - 1]->data;
          end = gru->dr[t]->end;
          while (p_dr < end) {
            float r = *p_r;
            *p_dr++ = *p_o1++ * *p_h++ * r * (1.f - r);
          }
          matrix_hadd_dot(grn_l->y1, gru->r[t], grn_l->y1);
          
          threaded_matmult(gru->dr[t], gru->tW_hr, grn_l->y3, NN, true, grnn->tp);
          p_dh = gru->dh[t - 1]->data;
          end = gru->dh[t - 1]->end;
          p_o1 = grn_l->y1->data;
          p_o3 = grn_l->y3->data;
          matrix_hadd_dot(gru->dh[t], gru->z[t], grn_l->y4);
          while (p_dh < end)
            *p_dh++ = *p_o1++ + *p_o2++ + *p_o3++ + *p_o4++;
        }
        if (grn_l->grad_in) {
          float *p_din = grn_l->grad_in[t]->data;
          end = grn_l->grad_in[t]->end;
          
          threaded_matmult(gru->dr[t], gru->tW_ir, grn_l->y1, NN, true, grnn->tp);
          threaded_matmult(gru->dz[t], gru->tW_iz, grn_l->y2, NN, true, grnn->tp);
          threaded_matmult(gru->dn[t], gru->tW_in, grn_l->y3, NN, true, grnn->tp);
          p_o1 = grn_l->y1->data;
          p_o2 = grn_l->y2->data;
          p_o3 = grn_l->y3->data;
          while (p_din < end)
            *p_din++ = *p_o1++ + *p_o2++ + *p_o3++;
        }
        transpose_by(grn_l->in[t],grn_l->t_i);
        threaded_matmult(grn_l->t_i, gru->dz[t], gru->dW_iz, NN, false,grnn->tp);
        threaded_matmult(grn_l->t_i, gru->dr[t], gru->dW_ir, NN, false,grnn->tp);
        threaded_matmult(grn_l->t_i, gru->dn[t], gru->dW_in, NN, false,grnn->tp);
        if (t > 0) {
          transpose_by(gru->h[t - 1],grn_l->t_g);
          threaded_matmult(grn_l->t_g, gru->dz[t], gru->dW_hz, NN, false, grnn->tp);
          threaded_matmult(grn_l->t_g, gru->dr[t], gru->dW_hr, NN, false, grnn->tp);
          p_r = gru->r[t]->data;
          end = gru->r[t]->end;
          float *p_h = gru->h[t - 1]->data;
          p_o1 = grn_l->y1->data;
          while (p_r < end)
            *p_o1++ = *p_r++ * *p_h++;
          transpose_by(grn_l->y1,grn_l->t_g);
          threaded_matmult(grn_l->t_g, gru->dn[t], gru->dW_hn, NN, false, grnn->tp);
        }
        matrix *delta = gru->dz[t];
        matrix *b_bff = gru->db_z;
        for (uint32_t r = 0; r < delta->row; r++)
          for (uint32_t c = 0; c < delta->col; c++)
            b_bff->data[c] += delta->data[r * delta->col + c];
        delta = gru->dr[t];
        b_bff = gru->db_r;
        for (uint32_t r = 0; r < delta->row; r++)
          for (uint32_t c = 0; c < delta->col; c++)
            b_bff->data[c] += delta->data[r * delta->col + c];
        delta = gru->dn[t];
        b_bff = gru->db_n;
        for (uint32_t r = 0; r < delta->row; r++)
          for (uint32_t c = 0; c < delta->col; c++)
            b_bff->data[c] += delta->data[r * delta->col + c];
            
        
      } break;
      }
    }
  }
  // clock_gettime(CLOCK_MONOTONIC, &t1);
  // sub_timespec(t0, t1, &delta_t);
  // printf("\nbackprop dt(s) = %d.%.4ld",(int)delta_t.tv_sec,delta_t.tv_nsec);
}

void update_grnnet(grnnet *grnn) {
  float max_norm = 5.0;
  for (uint32_t l = 0; l < grnn->n_layers; l++) {
    grnnet_layer *layer = grnn->layers + l;
    switch (layer->t_layer) {
    case GRU: {
      struct gru_layer *gru = &layer->layer.gru;
      SGD(gru->W_ir, gru->dW_ir, grnn->learn_rt, max_norm);
      SGD(gru->W_hr, gru->dW_hr, grnn->learn_rt, max_norm);
      SGD(gru->W_iz, gru->dW_iz, grnn->learn_rt, max_norm);
      SGD(gru->W_hz, gru->dW_hz, grnn->learn_rt, max_norm);
      SGD(gru->W_in, gru->dW_in, grnn->learn_rt, max_norm);
      SGD(gru->W_hn, gru->dW_hn, grnn->learn_rt, max_norm);
      SGD(gru->b_r, gru->db_r, grnn->learn_rt, max_norm);
      SGD(gru->b_z, gru->db_z, grnn->learn_rt, max_norm);
      SGD(gru->b_n, gru->db_n, grnn->learn_rt, max_norm);

      transpose_by(gru->W_ir,gru->tW_ir);
      transpose_by(gru->W_hr,gru->tW_hr);
      transpose_by(gru->W_iz,gru->tW_iz);
      transpose_by(gru->W_hz,gru->tW_hz);
      transpose_by(gru->W_in,gru->tW_in);
      transpose_by(gru->W_hn,gru->tW_hn);

    } break;
    case DENSE: {
      struct dense_layer *dense = &layer->layer.dense;
      SGD(dense->W, dense->dW, grnn->learn_rt, max_norm);
      SGD(dense->b, dense->db, grnn->learn_rt, max_norm);
    } break;
    }
  }
}

void run_grnnet(size_t epoch_max, grnnet *grnn, data_loader *dtl,
                STATE_RUN state, FILE *fout) {
  fputs("epoch,loss,acc\n", fout);
  puts("\nstart\n");

  matrix *out = grnn->layers[grnn->n_layers - 1].out[0];
  matrix *target = new_matrix(grnn->batch_size, out->col);

  struct timespec t0, t1, delta;
  float loss = 0, acc = 0, ba_md = (float)grnn->batch_size / dtl->size_lb;
  float lr_og = grnn->learn_rt;
  size_t end = dtl->n_itens / grnn->batch_size;
  for (size_t e = 0, i; e < epoch_max; e++) {
    loss = 0;
    acc = 0;

    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (i = 0; i < end; i++) {
      for (uint32_t t = 0; t < grnn->time_step; t++)
        load_batch_input(dtl, grnn->layers[0].in[t], grnn->time_step * t, i);

      load_batch_label(dtl, target, i);

      forward_grnnet(grnn);
      if (state == TRAIN) {
        backprop_grnnet(grnn, target);
        update_grnnet(grnn);
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

void train_grnnet(size_t epoch_max, grnnet *grnn, data_loader *dtl,
                  char *nmfile) {
  FILE *file_train = fopen(nmfile, "w");
  if (!file_train) {
    perror("\nERRO: nao foi possivel criar arquivo de treino.\n");
    exit(1);
  }
  run_grnnet(epoch_max, grnn, dtl, TRAIN, file_train);
  fclose(file_train);
}

void out_grnnet(grnnet *grnn, data_loader *dtl, char *namef) {
  FILE *file_test = fopen(namef, "w");
  if (!file_test) {
    perror("\nERRO: nao foi possivel criar arquivo de teste.\n");
    exit(1);
  }
  run_grnnet(1, grnn, dtl, TEST, file_test);
  fclose(file_test);
}
