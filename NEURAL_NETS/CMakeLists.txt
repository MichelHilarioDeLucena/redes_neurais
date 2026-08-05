cmake_minimum_required(VERSION 3.10)
project(neural_nets LANGUAGES C)

# Cria uma biblioteca com o código comum
add_library(common_lib STATIC
    data_loader/data_loader.c
    data_loader/mnist_driver.c
    mmath_lib/mm_lib.c
    mmath_lib/mm_threaded.c
)

# Link com biblioteca matemática
target_link_libraries(common_lib PRIVATE m)

# Flags para a biblioteca
target_compile_options(common_lib PRIVATE -O3 -march=native)
set_target_properties(common_lib PROPERTIES
    C_STANDARD 17
    C_STANDARD_REQUIRED ON
)

# --- MLP ---
add_executable(mlp mlp_test.c MLP/nnet.c)
target_link_libraries(mlp PRIVATE common_lib m)
target_compile_options(mlp PRIVATE -O3 -march=native)
set_target_properties(mlp PROPERTIES C_STANDARD 17 C_STANDARD_REQUIRED ON)

# --- CNN ---
add_executable(cnn cnn_test.c CNN/cnnet.c MLP/nnet.c )
target_link_libraries(cnn PRIVATE common_lib m)
target_compile_options(cnn PRIVATE -O3 -march=native)
set_target_properties(cnn PROPERTIES C_STANDARD 17 C_STANDARD_REQUIRED ON)

# --- RNN ---
add_executable(rnn rnn_test.c RNN/rnnet.c)
target_link_libraries(rnn PRIVATE common_lib m)
target_compile_options(rnn PRIVATE -O3 -march=native)
set_target_properties(rnn PROPERTIES C_STANDARD 17 C_STANDARD_REQUIRED ON)

# --- GRNN ---
add_executable(grnn grnn_test.c GRNN/grnnet.c)
target_link_libraries(grnn PRIVATE common_lib m)
target_compile_options(grnn PRIVATE -O3 -march=native)
set_target_properties(grnn PROPERTIES C_STANDARD 17 C_STANDARD_REQUIRED ON)
