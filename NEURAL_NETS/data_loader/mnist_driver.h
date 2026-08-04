#ifndef MNIST_INTERFACE
#define MNIST_INTERFACE

#include "data_loader.h"
#include <stdio.h>

typedef enum MNIST_CONFIG{
    IMG_HEADER_SIZE=16,
    LABEL_HEADER_SIZE=8,
    COLOR_MAX=255,
    MGC_LABEL=2049,
    MGC_IMAGE=2051
}MNIST_CONFIG;

typedef struct mnist_img_hder{
    uint32_t 
        magic_n,
        nitens,
        rows,
        cols;
}mnist_img_hder;

typedef struct mnist_lbl_hder{
    uint32_t 
        magic_n,
        nitens;
}mnist_lbl_hder;

data_loader * mnist_load(const char *in_path, const char *lbl_path);

int big_endian_read(void *dest,size_t size_data,size_t num_el ,FILE* file);
FILE * safe_get_mnist_file(const char *path);
void mnist_load_batch(data_loader *data,matrix *batch_input,matrix *batch_label,uint32_t pos);
void mnist_load_batch_input(data_loader *data,matrix *batch_input,size_t n_elem,uint32_t pos);
void mnist_load_batch_label(data_loader *data,matrix *batch_label,uint32_t pos);


#endif