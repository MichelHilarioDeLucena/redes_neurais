#include "mm_threaded.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>

#define FWR_ACTIVATION_DROPOUT(A_FUNC)                                         \
  for (; p_out < p_out_end; p_out += out_col) {                                \
    po_row = p_out;                                                            \
    float *pb = bias_data;                                                     \
    while (po_row < p_out + out_col) {                                         \
      *pz_out += *pb++;                                               \
      *po_row = A_FUNC(*pz_out++);                                             \
      float p_rand = rand_r(&rseed) / ((float)RAND_MAX);                       \
      if (p_rand < p_alive)                                                    \
        *p_mask++ = 1.0f, *po_row *= scale;                                    \
      else                                                                     \
        *p_mask++ = 0.0f, *po_row = 0.0f;                                      \
      po_row++;                                                                \
    }                                                                          \
  }
#define FWR_ACTIVATION_DROPOUT_LINEAR                                          \
  for (; p_out < p_out_end; p_out += out_col) {                                \
    po_row = p_out;                                                            \
    float *pb = bias_data;                                                     \
    while (po_row < p_out + out_col) {                                         \
      *pz_out +=*pb++;                                               \
      *po_row = *pz_out++;                                                     \
      float p_rand = rand_r(&rseed) / ((float)RAND_MAX);                       \
      if (p_rand < p_alive)                                                    \
        *p_mask++ = 1.0f, *po_row *= scale;                                    \
      else                                                                     \
        *p_mask++ = 0.0f, *po_row = 0.0f;                                      \
      po_row++;                                                                \
    }                                                                          \
  }

#define FWR_ACTIVATION(A_FUNC)                                                 \
  for (; p_out < p_out_end; p_out += out_col) {                                \
    po_row = p_out;                                                            \
    float *pb = bias_data, *end = p_out + out_col;                             \
    while (po_row < end) {                                                     \
      *pz_out += *pb++;                                                        \
      *po_row = A_FUNC(*pz_out++);                                             \
      po_row++;                                                                \
    }                                                                          \
  }

#define FWR_ACTIVATION_LINEAR                                                  \
  for (; p_out < p_out_end; p_out += out_col) {                                \
    po_row = p_out;                                                            \
    float *pb = bias_data;                                                     \
    while (po_row < p_out + out_col) {                                         \
      *pz_out += *pb++;                                                        \
      *po_row = *pz_out++;                                                     \
      po_row++;                                                                \
    }                                                                          \
  }

#define BCK_D_ACTIVATION_DROPOUT(D_A_FUNC)                                     \
  for (; p_gin < end; p_gin++, p_mask++)                                       \
    *p_gin = (*p_mask == 0.f) ? 0 : *p_gout++ * D_A_FUNC(*p_z++) * scale;

#define BCK_D_ACTIVATION(D_A_FUNC)                                             \
  for (; p_gin < end; p_gin++)                                                 \
    *p_gin = *p_gout++ * D_A_FUNC(*p_z++) * scale;

thread_pool *new_thread_pool() {
  thread_pool *tp = malloc(sizeof(thread_pool));
  tp->workers_count = sysconf(_SC_NPROCESSORS_ONLN);
  tp->active_workers = 0;
  tp->job_id = 0;
  tp->stop = 0;
  tp->workers = malloc(sizeof(pthread_t) * tp->workers_count);
  tp->tasks = malloc(sizeof(param) * tp->workers_count);

  pthread_mutex_init(&tp->lock, NULL);
  pthread_cond_init(&tp->cond_work, NULL);
  pthread_cond_init(&tp->cond_master, NULL);

  for (size_t i = 0; i < tp->workers_count; i++) {
    tp->tasks[i].id = i;
    tp->tasks[i].pool = tp;
    pthread_create(&tp->workers[i], NULL, kernel_operation, &tp->tasks[i]);
  }
  return tp;
}

void *kernel_operation(void *arg) {
  param *task = (param *)arg;
  thread_pool *tpool = task->pool;
  int local_job = 0;
  while (1) {
    pthread_mutex_lock(&tpool->lock);
    while (tpool->job_id == local_job && !tpool->stop)
      pthread_cond_wait(&tpool->cond_work, &tpool->lock);
    if (tpool->stop) {
      pthread_mutex_unlock(&tpool->lock);
      break;
    }
    local_job = tpool->job_id;
    pthread_mutex_unlock(&tpool->lock);

    switch (task->w_state) {
    case MATMULT: {
      KERNEL_MATRIX_MULT(
        task->state.matmult.a,
        task->state.matmult.b,
        task->state.matmult.c,
        task->start_row,
        task->end_row,
        task->state.matmult.type);
    } break;
    case F_LAYER_FUSION: {
      matrix *bias = task->state.f_fusion.a;
      matrix *mask = task->state.f_fusion.b;
      matrix *out = task->state.f_fusion.c;
      matrix *z_out = task->state.f_fusion.d;

      
      float *bias_data = bias->data;
      float *p_mask = mask->data + task->start_row * mask->col;
      float *pz_out = z_out->data + task->start_row * z_out->col;
      float *p_out = out->data + task->start_row * out->col, *po_row;

      float *p_out_end = out->data + task->end_row * out->col;
      float p_alive = task->state.f_fusion.p_alive;
      float scale = 1.f / p_alive;
      unsigned int rseed = task->state.f_fusion.rseed;
      size_t out_col = out->col;
      
      activ_func a_func_type = task->state.f_fusion.f_type;
      if(task->state.f_fusion.is_dropout)
        switch (a_func_type) {
        case SIGMOID: { FWR_ACTIVATION_DROPOUT(sigmoid)     } break;
        case TANH:    { FWR_ACTIVATION_DROPOUT(tanhf)       } break;
        case RELU:    { FWR_ACTIVATION_DROPOUT(ReLU)        } break;
        case L_RELU:  { FWR_ACTIVATION_DROPOUT(leaky_ReLU)  } break;
        case SILU:    { FWR_ACTIVATION_DROPOUT(silu)        } break;
        case LINEAR:
        case LOG_SOFTMAX:{
          FWR_ACTIVATION_LINEAR
        } break;
        }
      else{
        switch (a_func_type) {
        case SIGMOID: { FWR_ACTIVATION(sigmoid)     } break;
        case TANH:    { FWR_ACTIVATION(tanhf)       } break;
        case RELU:    { FWR_ACTIVATION(ReLU)        } break;
        case L_RELU:  { FWR_ACTIVATION(leaky_ReLU)  } break;
        case SILU:    { FWR_ACTIVATION(silu)        } break;
        case LINEAR:
        case LOG_SOFTMAX:{
          FWR_ACTIVATION_LINEAR
        } break;
        }
      }
    } break;
    case B_LAYER_FUSION: {
      matrix *z    = task->state.b_fusion.a;
      matrix *gin  = task->state.b_fusion.b;
      matrix *gout = task->state.b_fusion.c;
      matrix *mask = task->state.b_fusion.d;
      float *p_z    = z->data + task->start_row * z->col;
      float *p_gin  = gin->data + task->start_row * gin->col;
      float *p_gout = gout->data + task->start_row * gout->col;
      float *p_mask = mask->data + task->start_row * mask->col;

      float *end = gin->data + task->end_row * gin->col;
      float scale = 1.f / task->state.b_fusion.p_alive;
      activ_func da_func_type = task->state.b_fusion.df_type;
      if(task->state.b_fusion.is_dropout){ 
        switch (da_func_type) {
          case SIGMOID: {BCK_D_ACTIVATION_DROPOUT(d_sigmoid)    } break;
          case TANH:    {BCK_D_ACTIVATION_DROPOUT(d_tanh)       } break;
          case RELU:    {BCK_D_ACTIVATION_DROPOUT(d_ReLU)       } break;
          case L_RELU:  {BCK_D_ACTIVATION_DROPOUT(d_leaky_ReLU) } break;
          case SILU:    {BCK_D_ACTIVATION_DROPOUT(d_silu)       } break;
        }
      }else{
        switch (da_func_type) {
          case SIGMOID: {BCK_D_ACTIVATION(d_sigmoid)    } break;
          case TANH:    {BCK_D_ACTIVATION(d_tanh)       } break;
          case RELU:    {BCK_D_ACTIVATION(d_ReLU)       } break;
          case L_RELU:  {BCK_D_ACTIVATION(d_leaky_ReLU) } break;
          case SILU:    {BCK_D_ACTIVATION(d_silu)       } break;
          
        }
      }

    } break;
    }

    pthread_mutex_lock(&tpool->lock);
    if (!(--tpool->active_workers))
      pthread_cond_signal(&tpool->cond_master);
    pthread_mutex_unlock(&tpool->lock);
  }
  return NULL;
}

void threaded_matmult(matrix *a, matrix *b, matrix *c,type_matmult type,bool reset, thread_pool *tp) {
  
  uint32_t rows_slices =c->row / tp->workers_count;
  uint32_t rows_mod    =c->row  % tp->workers_count;

  if(reset)
    memset(c->data, 0, c->len * sizeof(float));

  pthread_mutex_lock(&tp->lock);
  for (uint32_t i = 0; i < tp->workers_count - 1; i++) {
    tp->tasks[i].state.matmult.a = a;
    tp->tasks[i].state.matmult.b = b;
    tp->tasks[i].state.matmult.c = c;
    tp->tasks[i].state.matmult.type = type;
    tp->tasks[i].w_state = MATMULT;

    tp->tasks[i].start_row = rows_slices * i;
    tp->tasks[i].end_row = rows_slices * (i + 1);
  }
  size_t last_i = tp->workers_count - 1;
  
  tp->tasks[last_i].state.matmult.a = a;
  tp->tasks[last_i].state.matmult.b = b;
  tp->tasks[last_i].state.matmult.c = c;
  tp->tasks[last_i].state.matmult.type = type;
  tp->tasks[last_i].w_state = MATMULT;
  tp->tasks[last_i].start_row = rows_slices * (tp->workers_count - 1);
  tp->tasks[last_i].end_row = rows_slices * tp->workers_count + rows_mod;

  tp->job_id++;
  tp->active_workers = tp->workers_count;
  pthread_cond_broadcast(&tp->cond_work);
  while (tp->active_workers > 0)
    pthread_cond_wait(&tp->cond_master, &tp->lock);
  pthread_mutex_unlock(&tp->lock);
}

void threaded_forward(matrix *bias, matrix *mask, matrix *out, matrix *z_out,
                      float p_val, int is_active, activ_func f_type,
                      thread_pool *tp) {

  uint32_t rows_slices = out->row / tp->workers_count;
  uint32_t rows_mod = out->row % tp->workers_count;

  pthread_mutex_lock(&tp->lock);

  for (uint32_t i = 0; i < tp->workers_count - 1; i++) {
    tp->tasks[i].state.f_fusion.a = bias;
    tp->tasks[i].state.f_fusion.b = mask;
    tp->tasks[i].state.f_fusion.c = out;
    tp->tasks[i].state.f_fusion.d = z_out;
    tp->tasks[i].state.f_fusion.p_alive = p_val;
    tp->tasks[i].state.f_fusion.rseed = rand();
    tp->tasks[i].state.f_fusion.f_type = f_type;

    tp->tasks[i].start_row = rows_slices * i;
    tp->tasks[i].end_row = rows_slices * (i + 1);
    tp->tasks[i].w_state = F_LAYER_FUSION;
    tp->tasks[i].state.f_fusion.is_dropout = is_active;
  }
  size_t last_i = tp->workers_count - 1;
  tp->tasks[last_i].state.f_fusion.a = bias;
  tp->tasks[last_i].state.f_fusion.b = mask;
  tp->tasks[last_i].state.f_fusion.c = out;
  tp->tasks[last_i].state.f_fusion.d = z_out;
  tp->tasks[last_i].state.f_fusion.p_alive = p_val;
  tp->tasks[last_i].state.f_fusion.rseed = rand();
  tp->tasks[last_i].state.f_fusion.f_type = f_type;

  tp->tasks[last_i].start_row = rows_slices * (tp->workers_count - 1);
  tp->tasks[last_i].end_row = rows_slices * tp->workers_count + rows_mod;
  tp->tasks[last_i].w_state = F_LAYER_FUSION;
  tp->tasks[last_i].state.f_fusion.is_dropout = is_active;

  tp->job_id++;
  tp->active_workers = tp->workers_count;

  pthread_cond_broadcast(&tp->cond_work);
  while (tp->active_workers > 0)
    pthread_cond_wait(&tp->cond_master, &tp->lock);
  pthread_mutex_unlock(&tp->lock);

  if(f_type==LOG_SOFTMAX)
    log_softmax(out);
}

void threaded_backward(matrix *z, matrix *gin, matrix *gout,matrix *mask, float p_val,int is_dropout,
                      activ_func d_func, thread_pool *tp){
  uint32_t rows_slices = z->row / tp->workers_count;
  uint32_t rows_mod = z->row % tp->workers_count;

  pthread_mutex_lock(&tp->lock);

  for (uint32_t i = 0; i < tp->workers_count - 1; i++) {
    tp->tasks[i].state.b_fusion.is_dropout = is_dropout;
    tp->tasks[i].state.b_fusion.a = z;
    tp->tasks[i].state.b_fusion.b = gin;
    tp->tasks[i].state.b_fusion.c = gout;
    tp->tasks[i].state.b_fusion.d = mask;
    tp->tasks[i].state.b_fusion.p_alive = p_val;
    tp->tasks[i].state.b_fusion.df_type = d_func;

    tp->tasks[i].start_row = rows_slices * i;
    tp->tasks[i].end_row = rows_slices * (i + 1);
    tp->tasks[i].w_state = B_LAYER_FUSION;
  }
  size_t last_i = tp->workers_count - 1;
  tp->tasks[last_i].state.b_fusion.is_dropout = is_dropout;
  tp->tasks[last_i].state.b_fusion.a = z;
  tp->tasks[last_i].state.b_fusion.b = gin;
  tp->tasks[last_i].state.b_fusion.c = gout;
  tp->tasks[last_i].state.b_fusion.d = mask;
  tp->tasks[last_i].state.b_fusion.p_alive = p_val;
  tp->tasks[last_i].state.b_fusion.df_type = d_func;

  tp->tasks[last_i].start_row = rows_slices * (tp->workers_count - 1);
  tp->tasks[last_i].w_state = B_LAYER_FUSION;
  tp->tasks[last_i].end_row = rows_slices * tp->workers_count + rows_mod;

  tp->job_id++;
  tp->active_workers = tp->workers_count;

  pthread_cond_broadcast(&tp->cond_work);
  while (tp->active_workers > 0)
    pthread_cond_wait(&tp->cond_master, &tp->lock);
  pthread_mutex_unlock(&tp->lock);
}

void destroy_thread_pool(thread_pool *tp) {
  pthread_mutex_lock(&tp->lock);
  tp->stop = 1;
  pthread_cond_broadcast(&tp->cond_work);
  pthread_mutex_unlock(&tp->lock);
  for (size_t i = 0; i < tp->workers_count; i++)
    pthread_join(tp->workers[i], NULL);
  free(tp->workers);
  free(tp->tasks);
  free(tp);
}

