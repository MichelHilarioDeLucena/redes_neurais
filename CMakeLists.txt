cmake_minimum_required(VERSION 3.10)
project(neural_nets LANGUAGES C)

set(SOURCES
    mlp_test.c                          # main para MLP
    MLP/nnet.c                          # ← caminho corrigido
    data_loader/data_loader.c           # ← caminho corrigido
    data_loader/mnist_driver.c          # ← caminho corrigido
    mmath_lib/mm_lib.c
    mmath_lib/mm_threaded.c
)

add_executable(mlp ${SOURCES})

target_compile_options(mlp PRIVATE -O3 -march=native)

set_target_properties(mlp PROPERTIES
    C_STANDARD 17
    C_STANDARD_REQUIRED ON
)
