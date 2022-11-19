#include "aisdk.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "example_utils.hpp"
#include <cstdio>
#include <chrono>
using namespace aisdk;



void test_Concat(){
    Env env = Env(device_type::CPU, 0);
    Tensor<float> input = Tensor<float>(&env, {1, 8});
    Tensor<float> input1 = Tensor<float>(&env, input, {1,4}, 0, {0.01, 0.02, 0.04, 0.1});
    Tensor<float> input2 = Tensor<float>(&env, input, {1,4}, 4, {0.3, 0.4, 0.01, 0.05});
    Tensor<float> output = Tensor<float>(&env, {1, 8});
    auto concat =  nn::op::Concat<float>(&env, {&input1, &input2}, output, 1);
    
    auto start = std::chrono::high_resolution_clock::now();
    int test_times = 1000;
    // for(int i = 0; i < test_times; ++i){
    //     concat_pm.execute(env.GetStream(), concat_args);
    // }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("Execution time: %ld microseconds", duration.count()/test_times);
    
}

void test_Conv()
{
    Env env = Env(device_type::CPU, 0);
    dims input_dims = {1, 1, 3, 3};
    dims output_dims = {1, 1, 3, 3};
    dims kernel_dims = {1, 1, 1, 1};
    dims stride = {1, 1};
    dims padding = {0, 0};
    Tensor<float> input = Tensor<float>(&env, input_dims);
    Tensor<float> output = Tensor<float>(&env, output_dims);
    input = {1., 1., 1. ,1., 1., 1., 1., 1., 1.};
    std::vector<float> weights = {1.};
    std::vector<float> bias = {1.};
    nn::layer::Conv<float> conv = nn::layer::Conv<float>(&env, input_dims, output_dims, kernel_dims, stride, padding, padding, weights, bias);
    int test_times = 1;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < test_times; ++i){
        conv.forward(input, output, true);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("Execution time: %ld microseconds", duration.count()/test_times);
    // printf("Conv Test:\n");
    // printf("Input:");
    // for(auto i : input.data){
    //     printf("%f ", i);
    // }
    // printf("\nOutput:");
    // for(auto i : output.data){
    //     printf("%f ", i);
    // }
    printf("\n");
}
void test_Linear1D(){
    Env env = Env(device_type::CPU, 0);
    dims input_dims = {1, 2};
    dims output_dims = {1, 8};
    Tensor<float> input = Tensor<float>(&env, input_dims);
    Tensor<float> output = Tensor<float>(&env, output_dims);
    Tensor<float> output1 = Tensor<float>(&env, {1,4});
    Tensor<float> output2 = Tensor<float>(&env, {1,4});
    output1.mem.set_data_handle(output.mem.get_data_handle());
    output2.mem.set_data_handle((float*)output.mem.get_data_handle() +  4);
    input = {0.0012, 0.03, 0.9, -0.9};
    nn::layer::Linear1D<float> linear1 = nn::layer::Linear1D<float>(&env, input_dims, {1, 4}, {0.001, 0.002, 0.003, 0.005, 0.001, 0.002, 0.005, 0.008}, {1, 1, 1, 1});
    nn::layer::Linear1D<float> linear2 = nn::layer::Linear1D<float>(&env, input_dims, {1, 4}, {0.021, 0.002, 0.003, 0.005, 0.001, 0.102, 0.0013, 0.038}, {2, 2, 2, 2});
    linear1.forward(input, output1, true);
    linear2.forward(input, output2, true);
    printf("Linear1D Test:\n");
    printf("Input:");
    for(auto i : input.data){
        printf("%f ", i);
    }
    printf("\nOutput1:");
    for(auto i : output1.data){
        printf("%f ", i);
    }
    printf("\nOutput2:");
    for(auto i : output2.data){
        printf("%f ", i);
    }
    printf("\n");
}

void test_ReLU(){
    Env env = Env(device_type::CPU, 0);
    dims dim = {1, 4};
    Tensor<float> input = Tensor<float>(&env, dim);
    Tensor<float> output = Tensor<float>(&env, dim);
    input = {0.0012, 0.03, 0.9, -0.9};
    nn::af::ReLU<float> relu = nn::af::ReLU<float>(&env, dim);
    relu.forward(input, output);
    output.save(output.data.data());
    printf("ReLU Test:\n");
    printf("Input:");
    for(auto i : input.data){
        printf("%f ", i);
    }
    printf("\nOutput:");
    for(auto i : output.data){
        printf("%f ", i);
    }
    printf("\n");
}



int main(){
    // test_ReLU();
    // test_Linear1D();
    test_Conv();
    return 0;
}