#ifndef MM_THREADED
#define MM_THREADED

#include "mm_lib.h"

#include <pthread.h>


typedef enum WORK_STATE { MATMULT, F_LAYER_FUSION, B_LAYER_FUSION } WORK_STATE;

typedef struct param {
  union state {
    struct {
      matrix *a, *b, *c;
      TYPE_MATMULT type;
      bool reset;
    } matmult;
    struct {
      matrix *a, *b, *c, *d;
      unsigned int rseed;
      AFUNC_TYPE f_type;
      float p_alive;
      int is_train;
    } f_fusion;
    struct {
      matrix *a, *b, *c;
      float p_alive;
      AFUNC_TYPE df_type;
    } b_fusion;
  } state;

  struct thread_pool *pool;
  uint32_t id, start_row, end_row;
  WORK_STATE w_state;

  int is_train;
} param;

typedef struct thread_pool {
  pthread_t *workers;
  pthread_mutex_t lock;
  pthread_cond_t cond_work;
  pthread_cond_t cond_master;
  int workers_count;
  int active_workers;
  int job_id;
  int stop;
  struct param *tasks;
} thread_pool;

thread_pool *new_thread_pool();
void *kernel_operation(void *arg);
void threaded_matmult(matrix *a, matrix *b, matrix *c,TYPE_MATMULT type,bool reset, thread_pool *tp);
void threaded_forward(matrix *bias, matrix *mask, matrix *out, matrix *z_out,
                      float p_val, int is_train, AFUNC_TYPE actv_type,
                      thread_pool *tp);
void threaded_backward(matrix *out, matrix *mask, matrix *delta, float p_val,
                      AFUNC_TYPE actv_type, thread_pool *tp);
void destroy_thread_pool(thread_pool *tp);


#endif