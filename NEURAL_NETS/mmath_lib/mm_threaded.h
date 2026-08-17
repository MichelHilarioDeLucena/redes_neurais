#ifndef MM_THREADED
#define MM_THREADED

#include "mm_lib.h"

#include <pthread.h>


typedef enum WORK_STATE { MATMULT, F_LAYER_FUSION, B_LAYER_FUSION } WORK_STATE;

typedef struct param {
  union state {
    struct {
      matrix *a, *b, *c;
      type_matmult type;
      bool reset;
    } matmult;
    struct {
      matrix *a, *b, *c, *d;
      unsigned int rseed;
      activ_func f_type;
      float p_alive;
      int is_dropout;
    } f_fusion;
    struct {
      matrix *a, *b, *c,*d;
      float p_alive;
      int is_dropout;
      activ_func df_type;
    } b_fusion;
  } state;

  struct thread_pool *pool;
  uint32_t id, start_row, end_row;
  WORK_STATE w_state;
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
void threaded_matmult(matrix *a, matrix *b, matrix *c,type_matmult type,bool reset, thread_pool *tp);
void threaded_forward(matrix *bias, matrix *mask, matrix *out, matrix *z_out,
                      float p_val, int is_train, activ_func actv_type,
                      thread_pool *tp);
void threaded_backward(matrix *z, matrix *gin, matrix *gout,matrix *mask, float p_val,int is_train,
                      activ_func dactv_type, thread_pool *tp);
void destroy_thread_pool(thread_pool *tp);


#endif