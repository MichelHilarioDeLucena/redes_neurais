#include "mm_lib.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

matrix *new_matrix(uint32_t row, uint32_t col) {
  matrix *new_m = malloc(sizeof(matrix));
  if(!new_m){
    perror("ERRO:falha em alocar mem.");
    exit(1);
  }
  *new_m = (matrix){row, col, row * col, calloc(row * col, sizeof(float))};
  new_m->end = new_m->data + new_m->len;
  return new_m;
}

matrix *new_matrix_set_data(uint32_t r, uint32_t c, float *data){
  matrix *new_m = malloc(sizeof(matrix));
  if(!new_m){
    perror("ERRO:falha em alocar mem.");
    exit(1);
  }
  *new_m = (matrix){r, c, r * c, data};
  new_m->end = new_m->data + new_m->len;
  return new_m;
}

tensor *new_tensor( uint32_t N, uint32_t H, uint32_t W, uint32_t C) {
  
  tensor *_tensor = calloc(1,sizeof(tensor));
  _tensor->N=N;
  _tensor->H=H;
  _tensor->W=W;
  _tensor->C=C;
  _tensor->len=N*H*W*C;
  _tensor->data = calloc(_tensor->len, sizeof(float));
  _tensor->data_end = _tensor->data + _tensor->len;
  return _tensor;
}

tensor *new_tensor_grad_init( uint32_t N, uint32_t H, uint32_t W, uint32_t C) {
  tensor *_tensor = new_tensor(N,H,W,C);
    _tensor->grad = calloc(_tensor->len, sizeof(float));
    _tensor->grad_end = _tensor->grad + _tensor->len;
  return _tensor;
}

void matrix_mult(matrix *restrict a, matrix *restrict b, matrix *restrict c,
                 type_matmult type, bool reset) {
  if (reset)
    memset(c->data, 0, c->len * sizeof(float));
  KERNEL_MATRIX_MULT(a, b, c, 0, c->row, type);
}

void transpose_by(matrix *a, matrix *t) {
  float *pa = a->data, *pa_col;
  float *pt = t->data;
  for (; pa < a->data + a->col; pa++)
    for (pa_col = pa; pa_col < a->end; pa_col += a->col, pt++)
      *pt = *pa_col;
}

void matrix_sum(matrix *a, matrix *b, matrix *c) {
  float *pa = a->data, *pb = b->data, *pc = c->data;
  float *end = pc + c->len;
  while (pc < end)
    *pc++ = *pa++ + *pb++;
}

void matrix_sub(matrix *a, matrix *b, matrix *c) {
  float *pa = a->data, *pb = b->data, *pc = c->data;
  float *end = pc + c->len;
  while (pc < end)
    *pc++ = *pa++ - *pb++;
}

void matrix_exp_sub(matrix *a, matrix *b, matrix *c) {
  float *pa = a->data, *pb = b->data, *pc = c->data;
  while (pc < c->end)
    *pc++ = expf(*pa++) - *pb++;
}

void matrix_hadd_dot_scalar(matrix *a, matrix *b, matrix *c, float k) {
  float *pa = a->data, *pb = b->data, *pc = c->data;
  float *end = pc + c->len;
  while (pc < end)
    *pc++ = (*pa++ * *pb++) * k;
}

void matrix_hadd_dot(matrix *a, matrix *b, matrix *c) {
  float *pa = a->data, *pb = b->data, *pc = c->data;
  float *end = pc + c->len;
  while (pc < end)
    *pc++ = *pa++ * *pb++;
}

void matrix_scalar_sum(matrix *a, float k) {
  float *pa = a->data;
  float *end = pa + a->len;
  for (; pa < end; pa++)
    *pa = *pa + k;
}

void matrix_sum_by_row(matrix *o, matrix *b) {
  float *po = o->data;
  float *pb = b->data;
  for (uint32_t r = 0; r < o->row; r++,pb++) {
    float *end_row = po + o->col;
    for (;po < end_row;po++) *po += *pb;
  }
}

void matrix_sum_by_col(matrix *o, matrix *b) {
  float *po = o->data;
  for (uint32_t r = 0; r < o->row; r++) {
    float *pb = b->data;
    float *end_row = po + o->col;
    for (;po < end_row;po++) *po += *pb++;
  }
}

void matrix_scalar_sub(matrix *a, float k) {
  float *pa = a->data;
  float *end = a->end;
  for (; pa < end; pa++)
    *pa = *pa - k;
}

void matrix_scalar_k_sub_b(matrix *a, float k,matrix *b) {
  float *pa = a->data,*pb=b->data;
  float *end = a->end;
  while (pa < end)
    *pa++ = k-*pb++;
}

void matrix_hadd_scalar_k_sub_b(matrix *a,matrix *b,matrix *c,float k){
  float *pa = a->data,*pb=b->data,*pc=c->data;
  float *end = a->end;
  while (pa < end)
    *pc++ = *pa++ * (k-*pb++);
}

void matrix_scalar_prod(matrix *a, float k) {
  float *pa = a->data;
  float *end = pa + a->len;
  for (; pa < end; pa++)
    *pa = *pa * k;
}

void matrix_to_tensor_NHWC(matrix *out_mat, tensor *output,uint32_t use_data){
  uint32_t N = output->N, H = output->H, W = output->W, C = output->C;
  float *op = out_mat->data;
  float *tp = use_data? output->data:output->grad;

  for (uint32_t n = 0; n < N; n++)
    for (uint32_t h = 0; h < H; h++)
      for (uint32_t w = 0; w < W; w++)
        for (uint32_t c = 0; c < C; c++) {
          uint32_t idx = c * (N * H * W) + (n * H + h) * W + w;
          uint32_t tidx = ((n * H + h) * W + w) * C + c;
          tp[tidx] = op[idx];
        }
}

void tensor_to_matrix_NHWC(matrix *out_mat, tensor *output,uint32_t use_data){
  uint32_t N = output->N, H = output->H, W = output->W, C = output->C;
  float *op = out_mat->data;
  float *tp = use_data? output->data:output->grad;

  for (uint32_t n = 0; n < N; n++)
    for (uint32_t h = 0; h < H; h++)
      for (uint32_t w = 0; w < W; w++)
        for (uint32_t c = 0; c < C; c++) {
          uint32_t idx = c * (N * H * W) + (n * H + h) * W + w;
          uint32_t tidx = ((n * H + h) * W + w) * C + c;
          op[idx] = tp[tidx];
        }
}

void SGD(matrix *theta, matrix *d_theta,float lr,float max_norm){
  float sum_sq = 0.0f;
  float *dt0 = d_theta->data;
  float *end_dt = d_theta->end;

  for (float *p = dt0; p < end_dt; p++)
    sum_sq += (*p) * (*p);
  float norm = sqrtf(sum_sq);
  float scale = 1.0f;
  if (norm > max_norm && norm > 1e-6f)
    scale = max_norm / norm;
  
  float *t0 = theta->data;
  float *end_t = theta->end;
  dt0 = d_theta->data;

  for (; t0 < end_t; t0++, dt0++)
    *t0 -= lr * (*dt0) * scale;
}

void init_uniform_distr(matrix *m, uint32_t i, uint32_t o) {
  float k = sqrtf(6.0f / (i + o));
  for (size_t w = 0; w < m->len; w++)
    m->data[w] = k * (2.0f * (rand() / (float)RAND_MAX) - 1.0f);
}

void init_uniform_distr_He(matrix *m, uint32_t i) {
  float k = sqrtf(6.0f / i);
  for (size_t w = 0; w < m->len; w++)
    m->data[w] = k * (2.0f * (rand() / (float)RAND_MAX) - 1.0f);
}

void init_uniform_distr_xors64(matrix *m, uint32_t i, uint32_t o,XorShift64State *rstate) {
  float k = sqrtf(6.0f / (i + o));
  for (size_t w = 0; w < m->len; w++)
    m->data[w] = k * (2.0f * xorshift_float(rstate) - 1.0f);
}

void init_uniform_distr_He_xors64(matrix *m, uint32_t i,XorShift64State *rstate) {
  float k = sqrtf(6.0f / i);
  for (size_t w = 0; w < m->len; w++)
    m->data[w] = k * (2.0f * xorshift_float(rstate) - 1.0f);
}

float mse(matrix *out, matrix t) {
  // matrix *out = net->outputs[net->n_layers - 1];
  float sum_err = 0, diff;
  for (size_t i = 0; i < out->len; i++) {
    diff = out->data[i] - t.data[i];
    sum_err += diff * diff;
  }
  return 0.5 * (sum_err / out->col);
}

inline float sigmoid(float z) { return 1.0 / (1 + expf(-z)); }

inline float d_sigmoid(float z) {
  float expz = expf(-z), plus_ez = 1.f + expz;
  return expz / (plus_ez * plus_ez);
}

inline float d_tanh(float z) {
  float _tanh = tanhf(z);
  return 1.f - _tanh * _tanh;
}

inline float ReLU(float z) { return z > 0 ? z : 0; }

inline float d_ReLU(float z) { return z > 0; }

inline float leaky_ReLU(float z) { return z > 0 ? z : z * 0.01f; }

inline float d_leaky_ReLU(float z) { return z > 0 ? 1.f : 0.01f; }

inline float silu(float z) { return z / (1.f + expf(-z)) + 0.01f * z; }

inline float d_silu(float z) {
  float expz = expf(-z);
  float ez = 1.f + expz;
  return (ez + z * expz) / (ez * ez) + 0.01f;
}

void ADAMW_correction(matrix *theta, matrix *mat_mw, matrix *mat_vw,
                      matrix *grad, float b1, float b2, float lr, uint64_t t,
                      float lambda) {

  float *mw = mat_mw->data;
  float *vw = mat_vw->data;
  float *gt = grad->data;
  float *w = theta->data;
  float den1 = 1.f / (1.f - powf(b1, t)), mt, vt;
  float den2 = 1.f / (1.f - powf(b2, t));

  float decay = 1.f - lambda * lr;
  float dif_b1 = 1.f - b1;
  float dif_b2 = 1.f - b2;
  // float clip=1.f;
  for (; mw < mat_mw->end; mw++, vw++, gt++, w++) {
    // if (*gt > clip)
    //   *gt = clip;
    // else if (*gt < -clip)
    //   *gt = clip;
    *mw = b1 * (*mw) + dif_b1 * (*gt);
    *vw = b2 * (*vw) + dif_b2 * (*gt) * (*gt);
    mt = *mw * den1;
    vt = *vw * den2;
    *w = *w * decay - lr * mt / (sqrtf(vt) + 1e-8f);
  }
}

void apply_dropout(matrix *out, matrix *mask, float p_alive) {
  float scale = 1.f / p_alive;
  float *p_out = out->data;
  float *p_mask = mask->data, p_rand;
  while (p_out < out->end) {
    p_rand = rand() / ((float)RAND_MAX);
    if (p_rand < p_alive)
      *p_mask++ = 1.0f, *p_out++ *= scale;
    else
      *p_mask++ = 0.0f, *p_out++ = 0.0f;
  }
}

void log_softmax(matrix *mat) {
  for (size_t r = 0; r < mat->row; r++) {
    float *row_data = mat->data + r * mat->col;
    float max_val = row_data[0];
    for (size_t c = 1; c < mat->col; c++) {
      if (row_data[c] > max_val) {
        max_val = row_data[c];
      }
    }
    float sum_exp = 0.0f;
    for (size_t c = 0; c < mat->col; c++) {
      row_data[c] -= max_val;
      sum_exp += expf(row_data[c]);
    }
    float log_sum = logf(sum_exp);
    for (size_t c = 0; c < mat->col; c++)
      row_data[c] -= log_sum;    
  }
}

float cat_cross_entropy(matrix *out, matrix *target) {
  float sum_err = 0;
  float *pout = out->data;
  float *ptar = target->data;
  for (; pout < out->data + out->len; pout++, ptar++)
    sum_err += (*ptar) * (*pout);
  return -sum_err / out->row;
}

float get_accuracy(matrix *y, matrix *t) {
  float sum = 0;
  for (uint32_t r = 0; r < y->row; r++) {
    float maxy = -INFINITY;
    float maxt = 0.0f;
    for (uint32_t c = 0; c < y->col; c++) {
      uint32_t idx = r * y->col + c;
      if (y->data[idx] > maxy) {
        maxy = y->data[idx];
        maxt = t->data[idx];
      }
    }
    sum += maxt;
  }
  
  return sum / y->row;
}

uint64_t xorshift64(XorShift64State *state) {
  uint64_t x = state->val;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  return state->val = x;
}

float xorshift_float(XorShift64State *state) {
    uint32_t val = (uint32_t)(xorshift64(state) >> 32);
    return val / MAX_32_INT_FLOAT; 
}

void destroy_matrix(matrix *mat) {
  free(mat->data);
  free(mat);
}

void sub_timespec(struct timespec t1, struct timespec t2, struct timespec *td) {
  td->tv_nsec = t2.tv_nsec - t1.tv_nsec;
  td->tv_sec = t2.tv_sec - t1.tv_sec;
  if (td->tv_sec > 0 && td->tv_nsec < 0) {
    td->tv_nsec += NS_PER_SECOND;
    td->tv_sec--;
  } else if (td->tv_sec < 0 && td->tv_nsec > 0) {
    td->tv_nsec -= NS_PER_SECOND;
    td->tv_sec++;
  }
}