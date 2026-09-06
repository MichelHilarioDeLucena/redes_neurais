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
  grnn->b1    = grn_params->b1;
  grnn->b2    = grn_params->b2;
  grnn->decay = grn_params->decay;
  grnn->n_layers = n_layers;
  grnn->time_step = t_step;
  grnn->adam_step = 1;
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
      grn_l->grad_in = (grn_l - 1)->grad_out;
    }
    
    switch (grn_l->t_layer) {
    case GRU: {
      grn_l->t_i = new_matrix(grn_l->in[0]->col,grn_l->in[0]->row);
      uint32_t h_size = scheme[l].config.gru.hidden_size;
      struct gru_layer *gru = &grn_l->layer.gru;
      grn_l->y1 = new_matrix(batch, h_size);
      grn_l->y2 = new_matrix(batch, h_size);
      grn_l->y3 = new_matrix(batch, h_size);
      grn_l->y4 = new_matrix(batch, h_size);
      grn_l->t_g = new_matrix(h_size, batch);
      
      gru->h = calloc(t_step, sizeof(matrix *));
      gru->dh = calloc(t_step, sizeof(matrix *));
      for (uint32_t t = 0; t < t_step; t++) {
        gru->h[t]  = new_matrix(batch, h_size);
        gru->dh[t] = new_matrix(batch, h_size);
      }

      uint32_t i_size = grn_l->in[0]->col;
      bool use_ln=scheme[l].config.gru.LN;
      gru->use_layer_norm = use_ln;
      init_gate(&gru->r,batch,h_size,i_size,t_step,use_ln);
      init_gate(&gru->z,batch,h_size,i_size,t_step,use_ln);
      init_gate(&gru->n,batch,h_size,i_size,t_step,use_ln);
      
      grn_l->out = gru->h;
      grn_l->grad_out = gru->dh;
    } break;
    case DENSE_GRNN: {
      struct dense_layer *dense = &grn_l->layer.dense;
      grnn_layer_t prev_type = scheme[l - 1].type;
      uint32_t out_size = scheme[l].config.dense.output_size;
      uint32_t in_size =
          (prev_type == GRU)     ? scheme[l - 1].config.gru.hidden_size
          : (prev_type == DENSE_GRNN) ? scheme[l - 1].config.dense.output_size
                                 : 0;
      grn_l->out = calloc(t_step, sizeof(matrix *));
      grn_l->grad_out = calloc(t_step, sizeof(matrix *));

      for (uint32_t t = 0; t < t_step; t++)
        grn_l->out[t] = new_matrix(batch, out_size),
        grn_l->grad_out[t] = new_matrix(batch, out_size);

      dense->W = new_matrix(in_size, out_size);
      dense->mW = new_matrix(in_size, out_size);
      dense->vW = new_matrix(in_size, out_size);
      init_uniform_distr(dense->W, in_size, out_size);
      dense->dW = new_matrix(in_size, out_size);
      dense->b = new_matrix(1, out_size);
      dense->mb = new_matrix(1, out_size);
      dense->vb = new_matrix(1, out_size);
      dense->db = new_matrix(1, out_size);
    } break;
    }
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
        threaded_matmult(grn_l->in[t], gru->z.Wi, gru->z.out[t], NN, true, grnn->tp);
        threaded_matmult(grn_l->in[t], gru->r.Wi, gru->r.out[t], NN, true, grnn->tp);
        if (t > 0) {
          threaded_matmult(gru->h[t - 1], gru->z.Wh, grn_l->y1, NN, true, grnn->tp);
          threaded_matmult(gru->h[t - 1], gru->r.Wh, grn_l->y2, NN, true, grnn->tp);
          matrix_sum(gru->z.out[t], grn_l->y1, gru->z.out[t]);
          matrix_sum(gru->r.out[t], grn_l->y2, gru->r.out[t]);
        }
        if(gru->use_layer_norm){
          layer_norm_forward(gru->z.out[t], gru->z.gamma, gru->z.b,
                       gru->z.mean[t], gru->z.var[t], gru->z.std_inv[t],
                       gru->z.x_hat[t], gru->z.epsilon);

          layer_norm_forward(gru->r.out[t], gru->r.gamma, gru->r.b,
                       gru->r.mean[t], gru->r.var[t], gru->r.std_inv[t],
                       gru->r.x_hat[t], gru->r.epsilon);
        }else{
          matrix_sum_by_col(gru->z.out[t], gru->z.b);
          matrix_sum_by_col(gru->r.out[t], gru->r.b);
        }
        
        float *end = gru->z.out[t]->end;
        float *p_z = gru->z.out[t]->data, *p_r = gru->r.out[t]->data;
        float *p_n = gru->n.out[t]->data, *p_h = gru->h[t]->data;
        for (; p_z < end; p_z++, p_r++)
        *p_z = sigmoid(*p_z), *p_r = sigmoid(*p_r);

        if (t > 0) {
          matrix_hadd_dot(gru->r.out[t], gru->h[t - 1], grn_l->y1);
          threaded_matmult(grn_l->y1, gru->n.Wh, grn_l->y2, NN, true, grnn->tp);
          threaded_matmult(grn_l->in[t], gru->n.Wi, grn_l->y1, NN, true, grnn->tp);
          matrix_sum(grn_l->y1, grn_l->y2, gru->n.out[t]);
        } else {
          threaded_matmult(grn_l->in[t], gru->n.Wi, gru->n.out[t], NN, true, grnn->tp);
        }
        if(gru->use_layer_norm){
          layer_norm_forward(gru->n.out[t], gru->n.gamma, gru->n.b,
                       gru->n.mean[t] , gru->n.var[t], gru->n.std_inv[t],
                       gru->n.x_hat[t], gru->n.epsilon);
        }else{
          matrix_sum_by_col(gru->n.out[t], gru->n.b);
        }

        for (end = gru->n.out[t]->end; p_n < end; p_n++)
          *p_n = tanhf(*p_n);
        matrix_scalar_k_sub_b(grn_l->y1, 1.f, gru->z.out[t]);
        if (t > 0) {
          matrix_hadd_dot(grn_l->y1, gru->n.out[t], grn_l->y2);
          matrix_hadd_dot(gru->z.out[t], gru->h[t - 1], grn_l->y1);
          matrix_sum(grn_l->y1, grn_l->y2, gru->h[t]);
        } else {
          matrix_hadd_dot(grn_l->y1, gru->n.out[t], gru->h[t]);
        }

      } break;

      case DENSE_GRNN: {
        struct dense_layer *dense = &grn_l->layer.dense;
        if (grnn->mode == MANY_TO_ONE && t == grnn->time_step - 1) {
          threaded_matmult(grn_l->in[t], dense->W, grn_l->out[0], NN, true,
                           grnn->tp);

          matrix_sum_by_col(grn_l->out[0], dense->b);
          log_softmax(grn_l->out[0]);
        }
        if (grnn->mode == MANY_TO_MANY) {
          threaded_matmult(grn_l->in[t], dense->W, grn_l->out[t], NN, true,
                           grnn->tp);
          matrix_sum_by_col(grn_l->out[t], dense->b);
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
      reset_gate(&gru->z);
      reset_gate(&gru->r);
      reset_gate(&gru->n);
    } break;
    case DENSE_GRNN: {
      struct dense_layer *dense = &grn_l->layer.dense;
      memset(dense->dW->data, 0, dense->dW->len * sizeof(float));
      memset(dense->db->data, 0, dense->db->len * sizeof(float));
    } break;
    }
  }
  for (int32_t t = grnn->time_step - 1; t >= 0; t--) {
    for (int32_t l = grnn->n_layers - 1; l >= 0; l--) {
      grnnet_layer *grn_l = grnn->layers + l;

      switch (grn_l->t_layer) {
      case DENSE_GRNN: {
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

        float *p_z = gru->z.out[t]->data, *end = gru->z.out[t]->end;
        float *p_n = gru->n.out[t]->data;
        float *p_dh = gru->dh[t]->data;
        float *p_dz = gru->z.dout[t]->data;
        float *p_dn = gru->n.dout[t]->data;

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
        if(gru->use_layer_norm){
          layer_norm_backward(
            gru->z.dout[t],
            gru->z.gamma,gru->z.std_inv[t],gru->z.x_hat[t],
            gru->z.dgamma,gru->z.db);
          layer_norm_backward(
            gru->n.dout[t],
            gru->n.gamma,gru->n.std_inv[t],gru->n.x_hat[t],
            gru->n.dgamma,gru->n.db);
        }else{
          matrix *delta = gru->z.dout[t];
          matrix *b_bff = gru->z.db;
          for (uint32_t r = 0; r < delta->row; r++)
            for (uint32_t c = 0; c < delta->col; c++)
              b_bff->data[c] += delta->data[r * delta->col + c];
          delta = gru->n.dout[t];
          b_bff = gru->n.db;
          for (uint32_t r = 0; r < delta->row; r++)
            for (uint32_t c = 0; c < delta->col; c++)
              b_bff->data[c] += delta->data[r * delta->col + c];
        }
        threaded_matmult(gru->z.dout[t], gru->z.tWh, grn_l->y2, NN, true, grnn->tp);
        threaded_matmult(gru->n.dout[t], gru->n.tWh, grn_l->y1, NN, true, grnn->tp);

        float *p_o1 = grn_l->y1->data;
        float *p_o2 = grn_l->y2->data;
        float *p_o4 = grn_l->y4->data, *p_o3;

        float *p_r = gru->r.out[t]->data;
        if (t > 0) {
          float *p_dr = gru->r.dout[t]->data;
          float *p_h = gru->h[t - 1]->data;
          end = gru->r.dout[t]->end;
          while (p_dr < end) {
            float r = *p_r;
            *p_dr++ = *p_o1++ * *p_h++ * r * (1.f - r);
          }
          if(gru->use_layer_norm){
            layer_norm_backward(
            gru->r.dout[t],            
            gru->r.gamma,gru->r.std_inv[t],gru->r.x_hat[t],
            gru->r.dgamma,gru->r.db);
          }else{
            matrix *delta = gru->r.dout[t];
            matrix *b_bff = gru->r.db;
            for (uint32_t r = 0; r < delta->row; r++)
              for (uint32_t c = 0; c < delta->col; c++)
                b_bff->data[c] += delta->data[r * delta->col + c];
          }
          matrix_hadd_dot(grn_l->y1, gru->r.out[t], grn_l->y1);
          
          threaded_matmult(gru->r.dout[t], gru->r.tWh, grn_l->y3, NN, true, grnn->tp);
          p_dh = gru->dh[t - 1]->data;
          end = gru->dh[t - 1]->end;
          p_o1 = grn_l->y1->data;
          p_o3 = grn_l->y3->data;
          matrix_hadd_dot(gru->dh[t], gru->z.out[t], grn_l->y4);
          while (p_dh < end)
            *p_dh++ = *p_o1++ + *p_o2++ + *p_o3++ + *p_o4++;
        }
        
        
        if (grn_l->grad_in) {
          float *p_din = grn_l->grad_in[t]->data;
          end = grn_l->grad_in[t]->end;
          
          threaded_matmult(gru->r.dout[t], gru->r.tWi, grn_l->y1, NN, true, grnn->tp);
          threaded_matmult(gru->z.dout[t], gru->z.tWi, grn_l->y2, NN, true, grnn->tp);
          threaded_matmult(gru->n.dout[t], gru->n.tWi, grn_l->y3, NN, true, grnn->tp);
          p_o1 = grn_l->y1->data;
          p_o2 = grn_l->y2->data;
          p_o3 = grn_l->y3->data;
          while (p_din < end)
            *p_din++ = *p_o1++ + *p_o2++ + *p_o3++;
        }
        transpose_by(grn_l->in[t],grn_l->t_i);
        threaded_matmult(grn_l->t_i, gru->z.dout[t], gru->z.dWi, NN, false,grnn->tp);
        threaded_matmult(grn_l->t_i, gru->r.dout[t], gru->r.dWi, NN, false,grnn->tp);
        threaded_matmult(grn_l->t_i, gru->n.dout[t], gru->n.dWi, NN, false,grnn->tp);
        if (t > 0) {
          transpose_by(gru->h[t - 1],grn_l->t_g);
          threaded_matmult(grn_l->t_g, gru->z.dout[t], gru->z.dWh, NN, false, grnn->tp);
          threaded_matmult(grn_l->t_g, gru->r.dout[t], gru->r.dWh, NN, false, grnn->tp);
          p_r = gru->r.out[t]->data;
          end = gru->r.out[t]->end;
          
          float *p_h = gru->h[t - 1]->data;
          p_o1 = grn_l->y1->data;
          while (p_r < end)
            *p_o1++ = *p_r++ * *p_h++;
          transpose_by(grn_l->y1,grn_l->t_g);
          threaded_matmult(grn_l->t_g, gru->n.dout[t], gru->n.dWh, NN, false, grnn->tp);
        }

      } break;
      }
    }
  }
}

void update_grnnet(grnnet *grnn) {
  float max_norm = 5.0;
  for (uint32_t l = 0; l < grnn->n_layers; l++) {
    grnnet_layer *layer = grnn->layers + l;
    switch (layer->t_layer) {
    case GRU: {
      struct gru_layer *gru = &layer->layer.gru;
      SGD_gate(&gru->z,grnn->learn_rt,max_norm);
      SGD_gate(&gru->r,grnn->learn_rt,max_norm);
      SGD_gate(&gru->n,grnn->learn_rt,max_norm);

    } break;
    case DENSE_GRNN: {
      struct dense_layer *dense = &layer->layer.dense;
      SGD(dense->W, dense->dW, grnn->learn_rt, max_norm);
      SGD(dense->b, dense->db, grnn->learn_rt, max_norm);
    } break;
    }
  }
}
void update_adamw_grnnet(grnnet *grnn) {
  float b1=grnn->b1,b2=grnn->b2;
  float lr=grnn->learn_rt;
  float decay=grnn->decay;
  uint64_t step=grnn->adam_step;
  for (uint32_t l = 0; l < grnn->n_layers; l++) {
    grnnet_layer *layer = grnn->layers + l;
    switch (layer->t_layer) {
    case GRU: {
      struct gru_layer *gru = &layer->layer.gru;
      if(gru->use_layer_norm){
        ADAMW_correction(gru->z.gamma,gru->z.mgamma,gru->z.vgamma,gru->z.dgamma,b1,b2,lr,step,decay);        
        ADAMW_correction(gru->r.gamma,gru->r.mgamma,gru->r.vgamma,gru->r.dgamma,b1,b2,lr,step,decay);
        ADAMW_correction(gru->n.gamma,gru->n.mgamma,gru->n.vgamma,gru->n.dgamma,b1,b2,lr,step,decay);
      }

      ADAMW_correction(gru->z.Wh,gru->z.mWh,gru->z.vWh,gru->z.dWh,b1,b2,lr,step,decay);
      ADAMW_correction(gru->r.Wh,gru->r.mWh,gru->r.vWh,gru->r.dWh,b1,b2,lr,step,decay);
      ADAMW_correction(gru->n.Wh,gru->n.mWh,gru->n.vWh,gru->n.dWh,b1,b2,lr,step,decay);

      ADAMW_correction(gru->z.Wi,gru->z.mWi,gru->z.vWi,gru->z.dWi,b1,b2,lr,step,decay);
      ADAMW_correction(gru->r.Wi,gru->r.mWi,gru->r.vWi,gru->r.dWi,b1,b2,lr,step,decay);
      ADAMW_correction(gru->n.Wi,gru->n.mWi,gru->n.vWi,gru->n.dWi,b1,b2,lr,step,decay);

      ADAMW_correction(gru->z.b,gru->z.mb,gru->z.vb,gru->z.db,b1,b2,lr,step,decay);
      ADAMW_correction(gru->r.b,gru->r.mb,gru->r.vb,gru->r.db,b1,b2,lr,step,decay);
      ADAMW_correction(gru->n.b,gru->n.mb,gru->n.vb,gru->n.db,b1,b2,lr,step,decay);

      transpose_by(gru->z.Wh, gru->z.tWh);
      transpose_by(gru->z.Wi, gru->z.tWi);
      transpose_by(gru->r.Wh, gru->r.tWh);
      transpose_by(gru->r.Wi, gru->r.tWi);
      transpose_by(gru->n.Wh, gru->n.tWh);
      transpose_by(gru->n.Wi, gru->n.tWi);
    } break;
    case DENSE_GRNN: {
      struct dense_layer *dense = &layer->layer.dense;
      ADAMW_correction(dense->W,dense->mW,dense->vW,dense->dW,b1,b2,lr,step,decay);
      ADAMW_correction(dense->b,dense->mb,dense->vb,dense->db,b1,b2,lr,step,decay);
    } break;
    }
  }
  grnn->adam_step++;
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
        // update_grnnet(grnn);
        update_adamw_grnnet(grnn);
      }
      loss += cat_cross_entropy(out, target);
      acc += get_accuracy(out, target) * 100.0;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    sub_timespec(t0, t1, &delta);
    printf("e = %ld | dt(s) = %d.%.4ld", e + 1, (int)delta.tv_sec, delta.tv_nsec);

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

void init_gate(gate *g, uint32_t B, uint32_t H, uint32_t I,uint32_t T,bool use_ln){
  g->epsilon = 1e-5f;
  g->Wh = new_matrix(H, H);
  g->mWh = new_matrix(H, H);
  g->vWh = new_matrix(H, H);
  g->Wi = new_matrix(I, H);
  g->mWi = new_matrix(I, H);
  g->vWi = new_matrix(I, H);
  g->tWh = new_matrix(H, H);
  g->tWi = new_matrix(H, I);
  g->b  = new_matrix(1, H);  
  g->mb  = new_matrix(1, H);  
  g->vb  = new_matrix(1, H);  
  g->dWh = new_matrix(H, H);
  g->dWi = new_matrix(I, H);
  g->db  = new_matrix(1, H);  
  g->out    = calloc(T, sizeof(matrix*)),
  g->dout   = calloc(T, sizeof(matrix*));
  
  for (uint32_t t = 0; t < T; t++)
    g->out    [t] = new_matrix(B, H),
    g->dout   [t] = new_matrix(B, H);

  if(use_ln){
    g->gamma = new_matrix(1, H);
    g->mgamma = new_matrix(1, H);
    g->vgamma = new_matrix(1, H);
    g->dgamma = new_matrix(1, H);
    g->mean   = calloc(T, sizeof(matrix*)),
    g->var    = calloc(T, sizeof(matrix*)),
    g->std_inv= calloc(T, sizeof(matrix*)),
    g->x_hat  = calloc(T, sizeof(matrix*));
    for (uint32_t t = 0; t < T; t++)
      g->mean   [t] = new_matrix(B, 1),
      g->var    [t] = new_matrix(B, 1),
      g->std_inv[t] = new_matrix(B, 1),
      g->x_hat  [t] = new_matrix(B, H);  
    for (int i = 0; i < H; i++) g->gamma->data[i] = 1.0f;
  }

  
  
  init_uniform_distr(g->Wi, I, H);
  init_uniform_distr(g->Wh, H, H);
  transpose_by(g->Wh, g->tWh);
  transpose_by(g->Wi, g->tWi);
}

void SGD_gate(gate *g,float lr,float maxg){
  SGD(g->Wi, g->dWi, lr, maxg);
  SGD(g->Wh, g->dWh, lr, maxg);
  SGD(g->b, g->db , lr, maxg);

  transpose_by(g->Wi,g->tWi);
  transpose_by(g->Wh,g->tWh);
}

void reset_gate(gate *g){  
  memset(g->dWi->data, 0, g->dWi->len * sizeof(float));
  memset(g->dWh->data, 0, g->dWh->len * sizeof(float));
  if(g->dgamma)
    memset(g->dgamma->data, 0, g->dgamma->len * sizeof(float));
  memset(g->db->data,  0, g->db->len * sizeof(float));
}

void destroy_gate(gate *g,uint32_t T,bool use_ln){
  destroy_matrix(g->Wh);
  destroy_matrix(g->Wi);
  destroy_matrix(g->tWh);
  destroy_matrix(g->tWi);
  destroy_matrix(g->b);
  destroy_matrix(g->dWh);
  destroy_matrix(g->dWi);
  destroy_matrix(g->db);
  for (uint32_t t = 0; t < T; t++)
    destroy_matrix(g->out [t]),
    destroy_matrix(g->dout[t]);
  free(g->out);
  free(g->dout);
  if(use_ln){
    destroy_matrix(g->gamma);
    destroy_matrix(g->dgamma);
    for (uint32_t t = 0; t < T; t++)
      destroy_matrix(g->mean   [t]),
      destroy_matrix(g->var    [t]),
      destroy_matrix(g->std_inv[t]),
      destroy_matrix(g->x_hat  [t]);
    free(g->mean   );
    free(g->var    );
    free(g->std_inv);
    free(g->x_hat  );
  }
  
}

void destroy_grnnet(grnnet *grnn){
  for (uint32_t l = 0; l < grnn->n_layers; l++) {
    grnnet_layer *grn_l = grnn->layers + l;
    switch (grn_l->t_layer){
    case GRU:{
      gru_layer *gru=&grn_l->layer.gru;
      for (uint32_t t = 0; t < grnn->time_step; t++)
        destroy_matrix(gru->h[t]),
        destroy_matrix(gru->dh[t]);
      free(gru->h);
      free(gru->dh);
      destroy_gate(&gru->z,grnn->time_step,gru->use_layer_norm);
      destroy_gate(&gru->r,grnn->time_step,gru->use_layer_norm);
      destroy_gate(&gru->n,grnn->time_step,gru->use_layer_norm);
      destroy_matrix(grn_l->y1 );
      destroy_matrix(grn_l->y2 );
      destroy_matrix(grn_l->y3 );
      destroy_matrix(grn_l->y4 );
      destroy_matrix(grn_l->t_g);
      destroy_matrix(grn_l->t_i);
      if (l == 0) {
        for (uint32_t t = 0; t < grnn->time_step; t++) {
            destroy_matrix(grn_l->in[t]);
            if (grn_l->grad_in) destroy_matrix(grn_l->grad_in[t]);
        }
        free(grn_l->in);
        if (grn_l->grad_in) free(grn_l->grad_in);
      }
    }break;
    case DENSE_GRNN:{
      struct dense_layer *dense = &grn_l->layer.dense;
      destroy_matrix(dense->W );
      destroy_matrix(dense->dW);
      destroy_matrix(dense->b );
      destroy_matrix(dense->db);
      for (uint32_t t = 0; t < grnn->time_step; t++) {
          destroy_matrix(grn_l->out[t]);
          destroy_matrix(grn_l->grad_out[t]);
      }
      free(grn_l->out);
      free(grn_l->grad_out);
    }break;
    }
  }
  free(grnn->layers);
  destroy_thread_pool(grnn->tp);
  free(grnn);
}
