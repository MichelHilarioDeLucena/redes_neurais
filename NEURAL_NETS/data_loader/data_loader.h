#ifndef DATA_LOADER
#define DATA_LOADER

#include "../mmath_lib/mm_lib.h"

typedef struct data_loader{
    uint32_t size_in,size_lb,stride_in,stride_lb,n_itens;
    void *data_in,*data_lb;
    void (*callback_load)(struct data_loader *dtl,matrix *batch_i,matrix *batch_l,uint32_t pos);
    void (*callback_load_input)(struct data_loader *dtl,matrix *batch_i,size_t n_elem,uint32_t pos);
    void (*callback_load_label)(struct data_loader *dtl,matrix *batch_l,uint32_t pos);
    void *aux_in,*aux_lbl;
    XorShift64State xors64_state;
}data_loader;

void shuffle_data(struct data_loader * dtl);
void load_batch(data_loader *dtl,matrix *batch_i,matrix *batch_l,uint32_t pos);
void load_batch_input(data_loader *dtl,matrix *batch_i,size_t n_elem,uint32_t pos);
void load_batch_label(data_loader *dtl,matrix *batch_l,uint32_t pos);
void destroy_loader(data_loader * dat_h);

#endif