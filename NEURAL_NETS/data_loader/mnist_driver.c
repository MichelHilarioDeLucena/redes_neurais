#include "mnist_driver.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int big_endian_read(void *dest, size_t size_data, size_t num_el, FILE *file) {
  byte *b_dest = dest;
  size_t end = num_el * size_data;
  size_t check = fread(b_dest, sizeof(byte), end, file);
  if (check != num_el * size_data) {
    perror("FALHA CRITICA: leitura corrompida do arquivo.\n");
    return 0;
  }
  size_t half_size = size_data >> 1;

  for (size_t j = 0; j < end; j += size_data) {
    size_t k = 2 * j + size_data - 1;
    size_t half = j + half_size;
    for (size_t i = j; i < half; i++) {
      byte swap = b_dest[i];
      b_dest[i] = b_dest[k - i];
      b_dest[k - i] = swap;
    }
  }
  return 1;
}

data_loader *mnist_load(const char *in_path, const char *lbl_path) {
  data_loader *dtl = malloc(sizeof(data_loader));
  FILE *fin = safe_get_mnist_file(in_path);
  FILE *flbl = safe_get_mnist_file(lbl_path);
  mnist_img_hder mih;
  mnist_lbl_hder mlh;

  big_endian_read(&mih, sizeof(uint32_t), 4, fin);
  if (mih.magic_n != MGC_IMAGE) {
    fprintf(stderr, "\nERRO: leitura de magic_number(img) %u != 2051.",
            mih.magic_n);
    exit(1);
  }
  big_endian_read(&mlh, sizeof(uint32_t), 2, flbl);
  if (mlh.magic_n != MGC_LABEL) {
    fprintf(stderr, "\nERRO: leitura de magic_number(label) %u != 2049.",
            mlh.magic_n);
    exit(1);
  }
  dtl->size_in = mih.nitens * mih.rows * mih.cols;
  dtl->data_in = malloc(dtl->size_in);
  dtl->stride_in = mih.rows * mih.cols;

  dtl->data_lb = malloc(mlh.nitens);
  dtl->size_lb = mlh.nitens;
  dtl->stride_lb = 1;

  if (!fread(dtl->data_in, 1, dtl->size_in, fin) ||
      !fread(dtl->data_lb, 1, dtl->size_lb, flbl)) {
    perror("ERRO NA LEITURA.\n");
    exit(1);
  }

  dtl->callback_load = mnist_load_batch;
  dtl->callback_load_input = mnist_load_batch_input;
  dtl->callback_load_label = mnist_load_batch_label;
  dtl->aux_in = malloc(dtl->stride_in);
  dtl->aux_lbl = malloc(dtl->stride_lb);
  dtl->xors64_state.val =
      (uint64_t)time(NULL) ^ (time(NULL) + 0xABCDEF123456789ULL);
  dtl->n_itens = dtl->size_lb / dtl->stride_lb;
  fclose(fin);
  fclose(flbl);
  return dtl;
}

FILE *safe_get_mnist_file(const char *path) {
  FILE *file_l = fopen(path, "rb");
  if (!file_l) {
    fprintf(stderr, "\nERRO: arquivo no caminho %s nao existe\n", path);
    exit(1);
  }
  return file_l;
}

void mnist_load_batch(data_loader *dtl, matrix *batch_input,
                      matrix *batch_label, uint32_t pos) {

  uint32_t bsize = batch_label->row;
  uint32_t num_classes = batch_label->col;
  memset(batch_label->data, 0, sizeof(float) * batch_label->len);

  byte *offset_lb = (byte *)dtl->data_lb + pos * dtl->stride_lb * bsize;

  for (uint32_t i = 0; i < bsize; i++)
    batch_label->data[i * num_classes + offset_lb[i]] = 1.0f;

  byte *offset_in = (byte *)dtl->data_in + pos * dtl->stride_in * bsize;
  float *p_in = batch_input->data;
  for (uint32_t i = 0; i < batch_input->len; i++) {
    float normalized = offset_in[i] / (float)COLOR_MAX;
    p_in[i] = (normalized - 0.1307f) / 0.3081f;
  }
}

void mnist_load_batch_input(data_loader *dtl, matrix *batch_input,
                            size_t n_elem, uint32_t pos) {
  uint32_t bsize = batch_input->row;
  float *p_in = batch_input->data;
  byte *offset_in = (byte *)dtl->data_in;
  for (size_t i = 0; i < bsize; i++){
    size_t _start = (i + pos * bsize) * dtl->stride_in + n_elem;
    for (size_t j = 0; j < batch_input->col; j++){
      float normalized = offset_in[_start + j] / (float)COLOR_MAX;
      *p_in++ = (normalized - 0.1307f) / 0.3081f;
    }
  }
}

void mnist_load_batch_label(data_loader *dtl, matrix *batch_label,
                            uint32_t pos) {
  uint32_t bsize = batch_label->row;
  uint32_t num_classes = batch_label->col;
  memset(batch_label->data, 0, sizeof(float) * batch_label->len);
  byte *offset_lb = (byte *)dtl->data_lb + pos * dtl->stride_lb * bsize;

  for (uint32_t i = 0; i < bsize; i++)
    batch_label->data[i * num_classes + offset_lb[i]] = 1.0f;
}