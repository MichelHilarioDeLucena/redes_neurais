#include "nnet.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

nnet *create_nnet(uint32_t scheme[], uint32_t count_lyr, uint32_t b_size,
                  uint32_t isize, AFUNC_TYPE func,thread_pool* tp,
                   float eta, float lambda, float p_alive, XorShift64State *rstate) {

  nnet *net = malloc(sizeof(nnet));

  *net = (nnet){.n_layers = count_lyr,
                .batch_size = b_size,
                .input_size = isize,
                .t_step 	= 1,
                .b1 		= .9f,
                .b2 		= .999f,
                .lambda    	= lambda,
                .p_alive   	= p_alive,
                .rand_state = rstate,
                .weights   	= malloc(sizeof(matrix *) * (count_lyr - 1)),
                .t_weights  = malloc(sizeof(matrix *) * (count_lyr - 1)),
                .bias      	= malloc(sizeof(matrix *) * (count_lyr - 1)),
                .gb_buffer 	= malloc(sizeof(matrix *) * (count_lyr - 1)),
                .outputs   	= malloc(sizeof(matrix *) * (count_lyr)),
                .t_outputs  = malloc(sizeof(matrix *) * (count_lyr)),

                .deltas    	= malloc(sizeof(matrix *) * (count_lyr - 1)),
                .z_out     	= malloc(sizeof(matrix *) * (count_lyr - 1)),
                .mask 	   	= malloc(sizeof(matrix *) * (count_lyr - 1)),
                .gw_buffer 	= malloc(sizeof(matrix *) * (count_lyr - 1)),
                .mw_adam   	= malloc(sizeof(matrix *) * (count_lyr - 1)),
                .vw_adam   	= malloc(sizeof(matrix *) * (count_lyr - 1)),
                .mb_adam   	= malloc(sizeof(matrix *) * (count_lyr - 1)),
                .vb_adam   	= malloc(sizeof(matrix *) * (count_lyr - 1)),
                .learn_rt 	= eta};

  net->a_func = func;
  net->t_outputs[0]=new_matrix(scheme[0], b_size);
  for (int i = 1; i < count_lyr; i++) {
    net->weights  [i - 1] = new_matrix(scheme[i - 1], scheme[i  ]);
    net->t_weights[i - 1] = new_matrix(scheme[i    ], scheme[i-1]);
    net->mw_adam  [i - 1] = new_matrix(scheme[i - 1], scheme[i  ]);
    net->vw_adam  [i - 1] = new_matrix(scheme[i - 1], scheme[i  ]);
    net->gw_buffer[i - 1] = new_matrix(scheme[i - 1], scheme[i  ]);
    if(rstate){
      if (func == RELU || func == L_RELU)
        init_uniform_distr_He_xors64(net->weights[i - 1], scheme[i -1],rstate);
      else
        init_uniform_distr_xors64(net->weights[i - 1], scheme[i - 1], scheme[i],rstate);
    }else{
      if (func == RELU || func == L_RELU)
        init_uniform_distr_He(net->weights[i - 1], scheme[i -1]);
      else
        init_uniform_distr(net->weights[i - 1], scheme[i - 1], scheme[i]);
    }

    transpose_by(net->weights[i - 1],net->t_weights[i - 1]);
    net->bias     [i - 1] = new_matrix(1, scheme[i]);
    net->gb_buffer[i - 1] = new_matrix(1, scheme[i]);
    net->mb_adam  [i - 1] = new_matrix(1, scheme[i]);
    net->vb_adam  [i - 1] = new_matrix(1, scheme[i]);

    net->z_out    [i - 1] = new_matrix(b_size, scheme[i]);
    net->outputs  [i    ] = new_matrix(b_size, scheme[i]);
    net->t_outputs[i    ] = new_matrix(scheme[i], b_size);

    net->deltas   [i - 1] = new_matrix(b_size, scheme[i]);
    net->mask     [i - 1] = new_matrix(b_size, scheme[i]);
  }
  net->input_grad=NULL;
  
  net->tp =!tp ? new_thread_pool() : tp;
  return net;
}

void forward_pass(matrix *input, nnet *net, STATE_RUN state) {
  net->outputs[0] = input;
  
  for (int l = 0; l < net->n_layers - 2; l++) {
    threaded_matmult(net->outputs[l], net->weights[l], net->outputs[l + 1],
                     NN,true,net->tp);
    threaded_forward(net->bias[l], net->mask[l], net->outputs[l + 1],
                     net->z_out[l], net->p_alive, state == TRAIN, net->a_func,
                     net->tp);
  }

  size_t last = net->n_layers - 2;
  matrix_mult(net->outputs[last], net->weights[last], net->outputs[last + 1],NN,true);
  matrix_sum_broadcast(net->outputs[last + 1], net->bias[last]);
  log_softmax(net->outputs[last + 1]);
}

void backprop(matrix *target, nnet *net) {
  size_t l_out = net->n_layers - 1;
  matrix_exp_sub(net->outputs[l_out], target, net->deltas[l_out - 1]);
  matrix_scalar_prod(net->deltas[l_out - 1], 1.f / net->batch_size);
  for (int l = l_out - 2; l >= 0; l--) {
    threaded_matmult(net->deltas[l + 1], net->t_weights[l+1], net->deltas[l],
                     NN,true,net->tp);
    threaded_backward(net->z_out[l], net->mask[l], net->deltas[l], net->p_alive,
                      net->a_func, net->tp);
    transpose_by(net->outputs[l],net->t_outputs[l]);
    threaded_matmult(net->t_outputs[l], net->deltas[l], net->gw_buffer[l], NN,true,net->tp);
    matrix_scalar_prod(net->gw_buffer[l], net->learn_rt);
  }
  if(net->input_grad)
    threaded_matmult(net->deltas[0], net->t_weights[0], net->input_grad,NN,true, net->tp);  
    
}

void update_layers(nnet *net) {
  for (size_t l = 0; l < net->n_layers - 1; l++) {
    matrix_sub(net->weights[l], net->gw_buffer[l], net->weights[l]);
    matrix *delta = net->deltas[l];
    matrix *b_bff = net->gb_buffer[l];
    for (uint32_t c = 0; c < delta->col; c++)
      b_bff->data[c] = 0.0f;
  
    for (uint32_t r = 0; r < delta->row; r++) {    
        for (uint32_t c = 0; c < delta->col; c++) {
            b_bff->data[c] += delta->data[r * delta->col + c];
        }
    }
    
    transpose_by(net->weights[l],net->t_weights[l]);
  }
}

void update_layers_adamw(nnet *net) {
  for (size_t l = 0; l < net->n_layers - 1; l++) {
    ADAMW_correction(net->weights[l], net->mw_adam[l], net->vw_adam[l],
                     net->gw_buffer[l], net->b1, net->b2, net->learn_rt,
                     net->t_step, net->lambda);

    matrix *delta = net->deltas[l];
    matrix *b_bff = net->gb_buffer[l];
    for (uint32_t c = 0; c < delta->col; c++)
      b_bff->data[c] = 0.0f;
    for (uint32_t r = 0; r < delta->row; r++){

      for (uint32_t c = 0; c < delta->col; c++)
      b_bff->data[c] += delta->data[r * delta->col + c];
    }
            
    ADAMW_correction(net->bias[l], net->mb_adam[l], net->vb_adam[l],
                     net->gb_buffer[l], net->b1, net->b2, net->learn_rt,
                     net->t_step, 0.f);
    transpose_by(net->weights[l],net->t_weights[l]);
  }
  net->t_step++;
}

void run_nnet(size_t epoch_max, nnet *net, data_loader *dtl, STATE_RUN state,
              FILE *fout) {
  fputs("epoch,error,acc\n", fout);
  puts("\nstart\n");

  matrix *batch_in = new_matrix(net->batch_size, net->input_size );
  uint32_t num_classes = net->outputs[net->n_layers - 1]->col; 
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
      load_batch(dtl, batch_in, target, i);

      forward_pass(batch_in, net, state);
      if (state == TRAIN) {
        backprop(target, net);
        // update_layers(net);
        update_layers_adamw(net);
      }
      error += cat_cross_entropy(net->outputs[net->n_layers - 1], target);
      acc += get_accuracy(net->outputs[net->n_layers - 1], target) * 100.0;
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
  destroy_matrix(batch_in);
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
  size_t check;
  if (file) {
    uint32_t len_l;
    if (!fread(&len_l, sizeof(uint32_t), 1, file))
      perror("falha ao ler.\n");
    for (int l = 0; l < len_l; l++) {
      check = 0;
      check += fread(&net->weights[l]->row, sizeof(uint32_t), 1, file);
      check += fread(&net->weights[l]->col, sizeof(uint32_t), 1, file);
      check += fread(net->weights[l]->data, sizeof(float), net->weights[l]->len,
                     file);
      check +=
          fread(net->bias[l]->data, sizeof(float), net->bias[l]->len, file);
      if (!check) {
        perror("falha:leitura corrompida.\n");
        break;
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
    uint32_t len_l = net->n_layers - 1;
    fwrite(&len_l, sizeof(int32_t), 1, file);
    for (int l = 0; l < len_l; l++) {
      fwrite(&net->weights[l]->row, sizeof(uint32_t), 1, file);
      fwrite(&net->weights[l]->col, sizeof(uint32_t), 1, file);
      fwrite(net->weights[l]->data, sizeof(float), net->weights[l]->len, file);
      fwrite(net->bias[l]->data, sizeof(float), net->bias[l]->len, file);
    }
  } else
    perror("ERRO ao salvar pesos.\n");
  fclose(file);
}

void destroy_nnet(nnet *net) {
  for (int i = 0; i < net->n_layers - 1; i++) {
    destroy_matrix(net->weights		[i]);
    destroy_matrix(net->gw_buffer	[i]);
    destroy_matrix(net->bias		[i]);
    destroy_matrix(net->outputs	[i + 1]);
    destroy_matrix(net->deltas		[i]);
    destroy_matrix(net->mask		[i]);
    destroy_matrix(net->gb_buffer	[i]);
    destroy_matrix(net->mb_adam		[i]);
    destroy_matrix(net->vb_adam		[i]);
    destroy_matrix(net->mw_adam		[i]);
    destroy_matrix(net->vw_adam		[i]);
  }
  free(net->weights);
  free(net->gw_buffer);
  free(net->bias);
  free(net->outputs);

  free(net->deltas);
  free(net->mask);
  free(net->gb_buffer);
  free(net->mw_adam);
  free(net->vw_adam);
  free(net->mb_adam);
  free(net->vb_adam);
  destroy_thread_pool(net->tp);
  free(net);
}
