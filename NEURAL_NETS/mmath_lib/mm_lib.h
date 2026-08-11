#ifndef MIMA_LIB
#define MIMA_LIB

#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#define byte unsigned char
#define MAX_32_INT_FLOAT 4294967295.0f

#define KERNEL_MATRIX_MULT(a, b, c, start_row, end_row, type)                  \
  float *pa,                                                                   \
      *pc, *pb;                                                                \
  switch (type) {                                                              \
  case NN: {                                                                   \
    pa = a->data + (start_row * a->col);                                       \
    pc = c->data + (start_row * c->col);                                       \
    float *c_end = c->data + (end_row * c->col);                               \
    for (; pc < c_end; pc += c->col, pa += a->col) {                           \
      pb = b->data;                                                            \
      float *end_pa = pa + a->col;                                             \
      for (float *pa_elem = pa, *pc_elem; pa_elem < end_pa; pa_elem++) {       \
        float *end_pc = pc + c->col,a_value=*pa_elem;                                           \
        for (pc_elem = pc; pc_elem < end_pc; pc_elem++, pb++)                  \
          *pc_elem += a_value * *pb;                                          \
      }                                                                        \
    }                                                                          \
  } break;                                                                     \
  case NT: {                                                                   \
    pa = a->data + (start_row * a->col);                                       \
    pc = c->data + (start_row * c->col);                                       \
    float *c_end = c->data + (end_row * c->col);                               \
    float *pa_elem;                                                            \
    for (; pc < c_end; pa += a->col)                                           \
      for (pb = b->data; pb < b->end; pc++) {                                  \
        float *end_pa = pa + a->col, sum = 0.f;                                \
        for (pa_elem = pa; pa_elem < end_pa; pa_elem++)                        \
          sum += *pa_elem * *pb++;                                             \
        *pc += sum;                                                            \
      }                                                                        \
  } break;                                                                     \
  case TN: {                                                                   \
    pa = a->data;                                                              \
    for (pb = b->data; pa < a->end; pa += a->col, pb += b->col) {              \
      float *end_pa = pa + end_row;                                            \
      pc = c->data + start_row * c->col;                                       \
      for (float *pa_elem = pa + start_row; pa_elem < end_pa; pa_elem++) {     \
        float *end_pc = pc + c->col;                                           \
        float *pb_elem = pb;                                                   \
        for (; pc < end_pc; pc++, pb_elem++)                                   \
          *pc += *pa_elem * *pb_elem;                                          \
      }                                                                        \
    }                                                                          \
  } break;                                                                     \
  case TT: {                                                                   \
    pa = a->data + start_row;                                                  \
    float *end_pa = a->data + end_row;                                         \
    pc = c->data + start_row * c->col;                                         \
    for (; pa < end_pa; pa++)                                                  \
      for (pb = b->data; pb < b->end; pc++) {                                  \
        float sum = 0.f;                                                       \
        for (float *pa_elem = pa; pa_elem < a->end; pa_elem += a->col)         \
          sum += *pa_elem * *pb++;                                             \
        *pc += sum;                                                            \
      }                                                                        \
                                                                               \
  } break;                                                                     \
  }

enum { NS_PER_SECOND = 1000000000 };

typedef enum AFUNC_TYPE { 
  L_RELU, RELU, SIGMOID, TANH, SILU,LOG_SOFTMAX, LINEAR
} AFUNC_TYPE;

typedef enum TYPE_MATMULT {
  NN,
  TN,
  NT,
  TT,
} TYPE_MATMULT;

typedef struct tensor {
  float *data, *data_end;
  float *grad, *grad_end;
  uint32_t N,H,W,C;
  size_t len;
} tensor;

typedef struct matrix {
  uint32_t row, col, len;
  float *restrict data, *restrict end;
} matrix;

typedef struct {
  uint64_t val;
} XorShift64State;

matrix *new_matrix(uint32_t r, uint32_t c);
matrix *new_matrix_set_data(uint32_t r, uint32_t c,float *data);

tensor *new_tensor( uint32_t N, uint32_t H, uint32_t W, uint32_t C);
tensor *new_tensor_grad_init( uint32_t N, uint32_t H, uint32_t W, uint32_t C);

void matrix_mult(matrix *restrict a, matrix *restrict b, matrix *restrict c,
                 TYPE_MATMULT type, bool reset);

void transpose_by(matrix *a, matrix *t);
void matrix_sum(matrix *a, matrix *b, matrix *c);
void matrix_sub(matrix *a, matrix *b, matrix *c);
void matrix_exp_sub(matrix *a, matrix *b, matrix *c);
void matrix_hadd_dot(matrix *a, matrix *b, matrix *c);
void matrix_hadd_dot_scalar(matrix *a, matrix *b, matrix *c, float k);

void matrix_scalar_sum(matrix *a, float k);
void matrix_sum_broadcast(matrix *o, matrix *b);
void matrix_scalar_sub(matrix *a, float k);
void matrix_scalar_k_sub_b(matrix *a, float k,matrix *b);
void matrix_hadd_scalar_k_sub_b(matrix *a,matrix *b,matrix *c,float k);
void matrix_scalar_prod(matrix *a, float k);
void SGD(matrix *theta, matrix *d_theta,float lr,float max_norm);

void init_uniform_distr(matrix *m, uint32_t i, uint32_t o);
void init_uniform_distr_xors64(matrix *m, uint32_t i, uint32_t o,XorShift64State *rstate);
void init_uniform_distr_He(matrix *m, uint32_t i);
void init_uniform_distr_He_xors64(matrix *m, uint32_t i,XorShift64State *rstate);
void log_softmax(matrix *mat);
void ADAMW_correction(matrix *weights, matrix *mw, matrix *vw, matrix *wgrad,
                      float b1, float b2, float lr, uint64_t t, float lambda);
void apply_dropout(matrix *out, matrix *mask, float p_alive);

float sigmoid(float z);
float d_sigmoid(float z);
float d_tanh(float z);
float ReLU(float z);
float d_ReLU(float z);
float leaky_ReLU(float z);
float d_leaky_ReLU(float z);
float silu(float z);
float d_silu(float z);
float mse(matrix *out, matrix t);

float cat_cross_entropy(matrix *out, matrix *target);
float get_accuracy(matrix *y, matrix *t);

uint64_t xorshift64(XorShift64State *state);
float xorshift_float(XorShift64State *state);
void destroy_matrix(matrix *mat);
void sub_timespec(struct timespec t1, struct timespec t2, struct timespec *td);
#endif
