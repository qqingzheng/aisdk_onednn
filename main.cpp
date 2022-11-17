#include "aisdk.h"
#include <cstdio>
#include <chrono>
using namespace aisdk;



void test_Concat(){
    Env env = Env(device_type::CPU, 0);
    dims_list input_dims_list = {{1,16},{1,16}};
    dims output_dims = {1, 32};
    Tensor<float> tensor1 = Tensor<float>(&env, input_dims_list[0], {0.03, 0.04, 0.05, 0.06, 0.03, 0.04, 0.05, 0.06, 0.03, 0.04, 0.05, 0.06, 0.03, 0.04, 0.05, 0.06});
    Tensor<float> tensor2 = Tensor<float>(&env, input_dims_list[1], {0.03, 0.04, 0.05, 0.06, 0.03, 0.04, 0.05, 0.06, 0.03, 0.04, 0.05, 0.06, 0.03, 0.04, 0.05, 0.06});
    Tensor<float> output = Tensor<float>(&env, output_dims);
    auto start = std::chrono::high_resolution_clock::now();
    int test_times = 100;
    for(int i = 0; i < test_times; ++i){
            nn::op::Concat<float>(&env, tensor1, tensor2, 1, output, true);
            env.wait();
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("Execution time: %ld microseconds", duration.count()/test_times);
    
}

void test_Linear1D(){
    Env env = Env(device_type::CPU, 0);
    dims input_dims = {1, 2};
    dims output_dims = {1, 4};
    Tensor<float> input = Tensor<float>(&env, input_dims);
    Tensor<float> output = Tensor<float>(&env, output_dims);
    input = {0.0012, 0.03, 0.9, -0.9};
    std::vector<float> weights = {0.001, 0.002, 0.003, 0.005, 0.001, 0.002, 0.005, 0.008};
    std::vector<float> bias = {1, 1, 1, 1};
    nn::layer::Linear1D<float> linear = nn::layer::Linear1D<float>(&env, input_dims, output_dims, weights, bias);
    linear.forward(input, output, true);
    printf("Linear1D Test:\n");
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
    test_Concat();
    return 0;
}