#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "nnet.h"
#include "../data_loader/data_loader.h"
#include "../data_loader/mnist_driver.h"

//gcc -O3 -march=native  main.c ../MLP/nnet.c ../mmath_lib/mm_lib.c ../mmath_lib/mm_threaded.c ../MLP/data_loader.c ../MLP/mnist_driver.c -lm -pthread  && ./a.out 80 0.001 15 120 120 10
//gcc -O3 -march=native main.c nnet.c data_loader.c mnist_drive.c ../mmath_lib/mm_lib.c ../mmath_lib/mm_threaded.c -o mlp
int main(int argc,char *argv[]){    
    size_t t=time(0);
    srand(t);    

    const char * train_path_labels="../archive_MNIST/train-labels.idx1-ubyte";
    const char * train_path_images="../archive_MNIST/train-images.idx3-ubyte";

    const char * test_path_labels="../archive_MNIST/t10k-labels.idx1-ubyte";
    const char * test_path_images="../archive_MNIST/t10k-images.idx3-ubyte";

    data_loader *data_ld=mnist_load(train_path_images,train_path_labels);

    size_t  batch_size,
            n_layer,
            epochs;
    float learn_rt;
    
    if(argc<=4){
        perror("parametros insuficientes.\n");
        return 1;
    }
        
    sscanf(argv[1],"%ld",&batch_size);
    sscanf(argv[2],"%f",&learn_rt);
    sscanf(argv[3],"%ld",&epochs);
    printf("%d\n",argc);
    n_layer=argc-3;
    
    uint32_t topology[n_layer];
    topology[0]=data_ld->stride_in;
    topology[n_layer-1]=10;
    XorShift64State rstate={.val=982173757522LL+time(0)};
    for(size_t i=1;i<n_layer-1;i++)
        sscanf(argv[i+3],"%d",topology+i);    
    for(size_t i=0;i<n_layer;i++)
        printf("tpl[%ld]:%u\n",i,topology[i]);
    nnet *mnist_net = create_nnet(
        topology,
        n_layer,
        batch_size,
        data_ld->stride_in,
        SILU,NULL,
        learn_rt,0, 1,&rstate);
        
    train_nnet(epochs,mnist_net,data_ld,"net_train.csv");
    // save_weights(mnist_net);
    destroy_loader(data_ld);
        
    // nnet_load(mnist_net,"net_weights1776885940.bin");
    data_ld=mnist_load(test_path_images,test_path_labels);
    
    // set_batch(mnist_net,100);
    out_nnet(mnist_net,data_ld,"net_test.csv");

    destroy_loader(data_ld);
    destroy_nnet(mnist_net);
    int result=0;
    result=system("python3 plot.py");
    return result;
}
