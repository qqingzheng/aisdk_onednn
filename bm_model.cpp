#include "aisdk.h"
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <fstream>
using namespace aisdk;
int main(){
    Env env = Env(device_type::CPU, 0);
    Tensor<float> pmi = Tensor<float>(&env, {1, 2}, {0., 0.});
    Tensor<float> pmi_feature = Tensor<float>(&env, {1, 8});
    Tensor<float> hv_feature = Tensor<float>(&env, {1, 32});

    Tensor<float> RI = Tensor<float>(&env, {1, 4}, {0., 1., 0., 0.});
    Tensor<float> RI_feature = Tensor<float>(&env, {1, 16});
    Tensor<float> CQI = Tensor<float>(&env, {1, 17}, {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.});
    Tensor<float> CQI_feature = Tensor<float>(&env, {1, 16});
    Tensor<float> TP = Tensor<float>(&env, {1, 17}, {0.13832, 0.13832, 0.13832, 0.13832, 0.13832, 0.13832, 0.13832, 0.13832, 0.13832, 0.13832, 0.13832, 0.13832, 0.13832, 0.13832, 0.13832, 0.13832, 0.13832});
    Tensor<float> TP_feature = Tensor<float>(&env, {1, 16});
    Tensor<float> layer_feature_concat = Tensor<float>(&env, {1, 48});
    Tensor<float> hidden_feature1 = Tensor<float>(&env, {1, 128});
    Tensor<float> hidden_feature2 = Tensor<float>(&env, {1, 256});

    Tensor<float> hv_layer_feature_concat = Tensor<float>(&env, {1, 288});
    Tensor<float> output = Tensor<float>(&env, {1, 18});

    // HV Feature Stream
    auto pmi_feature_layer = nn::layer::Linear1D_ReLU<float>(&env, {1, 2}, {1, 8}, "/home/lzj/aisdk_gui/model_bin/PMI_to_features_weight.bin", "/home/lzj/aisdk_gui/model_bin/PMI_to_features_bias.bin");
    auto hv_feature_layer = nn::layer::Linear1D_ReLU<float>(&env, {1, 8}, {1, 32}, "/home/lzj/aisdk_gui/model_bin/HV_features_weight.bin", "/home/lzj/aisdk_gui/model_bin/HV_features_bias.bin");
    // Layer Feature Stream
    auto RI_feature_layer = nn::layer::Linear1D<float>(&env, {1, 4}, {1, 16}, "/home/lzj/aisdk_gui/model_bin/RI_to_features_weight.bin", "/home/lzj/aisdk_gui/model_bin/RI_to_features_bias.bin");
    auto CQI_feature_layer = nn::layer::Linear1D<float>(&env, {1, 17}, {1, 16}, "/home/lzj/aisdk_gui/model_bin/CQI_to_features_weight.bin", "/home/lzj/aisdk_gui/model_bin/CQI_to_features_bias.bin");
    auto TP_feature_layer = nn::layer::Linear1D<float>(&env, {1, 17}, {1, 16}, "/home/lzj/aisdk_gui/model_bin/TP_to_features_weight.bin", "/home/lzj/aisdk_gui/model_bin/TP_to_features_bias.bin");
    auto hidden_feature_layer1 = nn::layer::Linear1D_ReLU<float>(&env, {1, 48}, {1, 128}, "/home/lzj/aisdk_gui/model_bin/layer_hiden_layer1_weight.bin", "/home/lzj/aisdk_gui/model_bin/layer_hiden_layer1_bias.bin");
    auto hidden_feature_layer2 = nn::layer::Linear1D_ReLU<float>(&env, {1, 128}, {1, 256}, "/home/lzj/aisdk_gui/model_bin/layer_hiden_layer2_weight.bin", "/home/lzj/aisdk_gui/model_bin/layer_hiden_layer2_bias.bin");
    auto joint_layer = nn::layer::Linear1D<float>(&env, {1, 288}, {1, 18}, "/home/lzj/aisdk_gui/model_bin/joint_layer_weight.bin", "/home/lzj/aisdk_gui/model_bin/joint_layer_bias.bin");
    
    auto start = std::chrono::high_resolution_clock::now();
    int test_times = 1;
    for(int i = 0; i < test_times; ++i){
        pmi_feature_layer.forward(pmi, pmi_feature);
        hv_feature_layer.forward(pmi_feature, hv_feature);
        RI_feature_layer.forward(RI, RI_feature);
        CQI_feature_layer.forward(CQI, CQI_feature);
        TP_feature_layer.forward(TP, TP_feature);
        nn::op::Concat<float>(&env, {&RI_feature, &CQI_feature, &TP_feature}, 1, layer_feature_concat);
        hidden_feature_layer1.forward(layer_feature_concat, hidden_feature1);
        hidden_feature_layer2.forward(hidden_feature1, hidden_feature2);
        nn::op::Concat<float>(&env, hidden_feature2, hv_feature, 1, hv_layer_feature_concat);
        joint_layer.forward(hv_layer_feature_concat, output, true);
    }
    env.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("Execution time: %ld microseconds", duration.count()/test_times);
    // output.save();
    // for(auto value : output.data){
    //     printf("%f ", value);
    // }
    return 0;
}