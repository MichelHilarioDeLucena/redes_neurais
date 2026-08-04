#include "data_loader.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <limits.h>

void shuffle_data(data_loader *dtl){
    uint64_t itens= dtl->size_lb/dtl->stride_lb;
    byte *aux_i=dtl->aux_in;
    byte *aux_lbl=dtl->aux_lbl;
    byte *din=dtl->data_in;
    byte *dlbl=dtl->data_lb;
    for(uint64_t i=itens-1;i>0;i--){
        uint64_t r = xorshift64(&dtl->xors64_state) % (i + 1);
        memcpy(aux_i,din+i*dtl->stride_in,dtl->stride_in);
        memcpy(din+i*dtl->stride_in,din+r*dtl->stride_in,dtl->stride_in);
        memcpy(din+r*dtl->stride_in,aux_i,dtl->stride_in);
        
        memcpy(aux_lbl,dlbl+i*dtl->stride_lb,dtl->stride_lb);
        memcpy(dlbl+i*dtl->stride_lb,dlbl+r*dtl->stride_lb,dtl->stride_lb);
        memcpy(dlbl+r*dtl->stride_lb,aux_lbl,dtl->stride_lb);
        
    }
}

void load_batch(data_loader *dtl,matrix *batch_i,matrix *batch_l,uint32_t pos){
    dtl->callback_load(dtl,batch_i,batch_l,pos);
}
void load_batch_input(data_loader *dtl,matrix *batch_i,size_t nlem,uint32_t pos){
    dtl->callback_load_input(dtl,batch_i,nlem,pos);
}
void load_batch_label(data_loader *dtl,matrix *batch_l,uint32_t pos){
    dtl->callback_load_label(dtl,batch_l,pos);
}

void destroy_loader(data_loader * dtl){
    free(dtl->data_in);
    free(dtl->data_lb);
    free(dtl->aux_in);
    free(dtl->aux_lbl);
    free(dtl);
}