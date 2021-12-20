cmake_minimum_required(VERSION 3.20)
project(MNIST_encoder)

set(CMAKE_CXX_STANDARD 14)

find_package(ROOT REQUIRED COMPONENTS Core Gui)

include_directories(matrix)
include_directories(neural_net)
include_directories(ROOT_INCLUDE_DIRS)


set(matrix_dir
        matrix/matrix.h
        matrix/matrix.cpp)

set(neural_net_dir
        neural_net/layer.h
        neural_net/layer.cpp
        neural_net/neural_net.h
        neural_net/neural_net.cpp
        neural_net/linear_layer.h
        neural_net/linear_layer.cpp
        )

add_executable(mnist-image-recognition-train examples/MNIST-image-recognition/mnist.cpp
        ${matrix_dir}
        ${neural_net_dir})
add_executable(mnist-recognition-test examples/MNIST-image-recognition/mnist_test.cpp
        ${matrix_dir}
        ${neural_net_dir}
        )
add_executable(mnist-encoder-train examples/MNIST-encoder/encoder-train.cpp
        ${matrix_dir}
        ${neural_net_dir}
        )
add_executable(mnist-encoder-test examples/MNIST-encoder/encoder-test.cpp
        ${matrix_dir}
        ${neural_net_dir}
        )


set(executables
        mnist-image-recognition-train
        mnist-recognition-test
        mnist-encoder-train
        mnist-encoder-test
        )

foreach (exe IN LISTS executables)
    target_link_libraries(${exe} PUBLIC ROOT::Core ROOT::Gui)
endforeach ()
