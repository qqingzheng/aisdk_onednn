#include "nn/absrtactnn.hpp"
#include "data/tensor.hpp"
#include <iostream>
#include <fstream>
// #include "oneapi/dnnl/dnnl_types.h"
using namespace dnnl;

namespace aisdk{ namespace nn{

    namespace utils{
        template<typename T>
        class Sequential{

        };
    }

    namespace op{
        template<typename T>
        class Concat{
            private:
                Env* env;
                concat primitive;
                std::unordered_map<int, memory> concat_args;
            public:
                Concat(Env* env_ptr, const std::vector<Tensor<T>*> tensors, const Tensor<T>& dst, int axis){
                    env = env_ptr;
                    std::vector<memory::desc> src_mds;
                    for(Tensor<T>* tensor : tensors){
                        src_mds.push_back(tensor->shape.memory_desc);
                    }
                    auto concat_pd = concat::primitive_desc(env_ptr->GetEngine(), dst.shape.memory_desc, axis, src_mds);
                    primitive = concat(concat_pd);
                    for(int i = 0; i < tensors.size(); ++ i){
                        concat_args.insert({DNNL_ARG_MULTIPLE_SRC + i, tensors[i]->mem});
                    }
                    
                }
                void forward(Tensor<T>& output, bool save_process = false){
                    concat_args.insert({DNNL_ARG_DST, output.mem});
                    primitive.execute(env->GetStream(), concat_args);
                    if(save_process){
                        env->GetStream().wait();
                        output.save();
                    }
                }
        };
    }

    namespace af{
        template<typename T>
        class ReLU : Layer<T>{
            private:
                Env* env;
                eltwise_forward primitive;
                eltwise_forward::primitive_desc pd;
                Shape<T> md;
            public:
                ReLU(Env* env_ptr, const dims& dim, float alpha, float beta){
                    this->env = env_ptr;
                    md = Shape<T>(env, dim);
                    pd = eltwise_forward::primitive_desc(env->GetEngine(),
                            prop_kind::forward_inference, algorithm::eltwise_relu,
                            md.memory_desc, 
                            md.memory_desc, 
                            alpha, 
                            beta 
                        
                    );
                    primitive = eltwise_forward(pd);
                }
                ReLU(Env* env_ptr, const dims& dim){
                    this->env = env_ptr;
                    md = Shape<T>(env, dim);
                    pd = eltwise_forward::primitive_desc(env->GetEngine(),
                            prop_kind::forward_inference, algorithm::eltwise_relu,
                            md.memory_desc, 
                            md.memory_desc,
                            0.f, 
                            0.f
                    );
                    primitive = eltwise_forward(pd);
                }
                void forward(Tensor<T>& input, Tensor<T>& output, bool save_process = false){
                    primitive.execute(env->GetStream(),
                    {
                        {DNNL_ARG_SRC, input.mem},
                        {DNNL_ARG_DST, output.mem}
                    });
                    if(save_process){
                        env->GetStream().wait();
                        output.save();
                    }
                }

        };
    }

    namespace layer{
        template<typename T>
        class Linear1D : Layer<T>{
            private:
                Env* env;
                Shape<T> md;
                bool is_weight_reorder = false;
                bool first_run = true;
                std::unordered_map<int, memory> inner_product_args;
                Tensor<T> reordered_weight_data;
                Tensor<T> weights_data;
                Tensor<T> bias_data;
                inner_product_forward primitive;
                dims input_dims;
                dims output_dims;
            public:
                Linear1D(Env* env_ptr, dims input_dims, dims output_dims, const char* weights_file, const char* bias_file){
                    this->env = env_ptr;
                    this->input_dims = input_dims;
                    this->output_dims = output_dims;  
                    weights_data.init(env, {output_dims[1], input_dims[1]});
                    bias_data.init(env, {output_dims[1]});
                    reordered_weight_data.init_nomem(env, {output_dims[1], input_dims[1]}, true);
                    std::vector<T> weights(weights_data.shape.size);
                    std::vector<T> bias(bias_data.shape.size);
                    std::ifstream weight_in;
                    std::ifstream bias_in;
                    weight_in.open(weights_file, std::ios::in | std::ios::binary);
                    weight_in.read((char*)weights.data(),  sizeof(T)*weights_data.shape.size);
                    weight_in.close();
                    bias_in.open(bias_file, std::ios::in | std::ios::binary);
                    bias_in.read((char*)bias.data(),  sizeof(T)*bias_data.shape.size);
                    bias_in.close();
                    init(weights, bias);
                }
                Linear1D(Env* env_ptr, dims input_dims, dims output_dims){
                    this->env = env_ptr;
                    this->input_dims = input_dims;
                    this->output_dims = output_dims;  
                    weights_data.init(env, {output_dims[1], input_dims[1]});
                    bias_data.init(env, {output_dims[1]});
                    reordered_weight_data.init_nomem(env, {output_dims[1], input_dims[1]}, true);
                }
                Linear1D(Env* env_ptr, dims input_dims, dims output_dims, const std::vector<T>& weights, const std::vector<T>& bias){
                    this->env = env_ptr;
                    this->input_dims = input_dims;
                    this->output_dims = output_dims;  
                    weights_data.init(env, {output_dims[1], input_dims[1]});
                    bias_data.init(env, {output_dims[1]});
                    reordered_weight_data.init_nomem(env, {output_dims[1], input_dims[1]}, true);
                    init(weights, bias);
                }
                void init(const std::vector<T>& weights, const std::vector<T>& bias){
                    SetWeight(weights);
                    SetBias(bias);
                    Shape<T> src_shape = Shape<T>(env, input_dims);
                    Shape<T> dst_shape = Shape<T>(env, output_dims);
                    inner_product_forward::primitive_desc linear_pd = inner_product_forward::primitive_desc(
                            env->GetEngine(), prop_kind::forward_training, src_shape.memory_desc, reordered_weight_data.shape.memory_desc, bias_data.shape.memory_desc, dst_shape.memory_desc);

                    bool need_reorder_weights = linear_pd.weights_desc() != weights_data.shape.memory_desc;
                    if (need_reorder_weights) {
                        is_weight_reorder = true;
                        reordered_weight_data.mem = memory(linear_pd.weights_desc(), env->GetEngine());
                        reorder(weights_data.mem, reordered_weight_data.mem).execute(env->GetStream(), weights_data.mem, reordered_weight_data.mem);
                    }
                    
                    primitive = inner_product_forward(linear_pd);
                }
                void SetWeight(const std::vector<T>& weights){
                    this->weights_data = weights;
                }
                void SetBias(const std::vector<T>& bias){
                    this->bias_data = bias;
                }
                void forward(Tensor<T>& input, Tensor<T>& output, bool save_process = false){
                    if(first_run){
                        inner_product_args.insert({DNNL_ARG_SRC, input.mem});
                        if(is_weight_reorder){
                            inner_product_args.insert({DNNL_ARG_WEIGHTS, reordered_weight_data.mem});
                        }else{
                            inner_product_args.insert({DNNL_ARG_WEIGHTS, weights_data.mem});
                        }
                        inner_product_args.insert({DNNL_ARG_BIAS, bias_data.mem});
                        inner_product_args.insert({DNNL_ARG_DST, output.mem});
                        first_run = false;
                    }
                    primitive.execute(env->GetStream(), inner_product_args);
                    if(save_process){
                        env->GetStream().wait();
                        output.save();
                    }
                }

        };
        template<typename T>
        class Linear1D_ReLU : Layer<T>{
            private:
                Env* env;
                Shape<T> md;
                bool is_weight_reorder = false;
                bool first_run = true;
                std::unordered_map<int, memory> inner_product_args;
                Tensor<T> reordered_weight_data;
                Tensor<T> weights_data;
                Tensor<T> bias_data;
                inner_product_forward primitive;
                dims input_dims;
                dims output_dims;
            public:
                Linear1D_ReLU(Env* env_ptr, dims input_dims, dims output_dims, const char* weights_file, const char* bias_file){
                    this->env = env_ptr;
                    this->input_dims = input_dims;
                    this->output_dims = output_dims;  
                    weights_data.init(env, {output_dims[1], input_dims[1]});
                    bias_data.init(env, {output_dims[1]});
                    reordered_weight_data.init_nomem(env, {output_dims[1], input_dims[1]}, true);
                    std::vector<T> weights(weights_data.shape.size);
                    std::vector<T> bias(bias_data.shape.size);
                    std::ifstream weight_in;
                    std::ifstream bias_in;
                    weight_in.open(weights_file, std::ios::in | std::ios::binary);
                    weight_in.read((char*)weights.data(),  sizeof(T)*weights_data.shape.size);
                    weight_in.close();
                    bias_in.open(bias_file, std::ios::in | std::ios::binary);
                    bias_in.read((char*)bias.data(),  sizeof(T)*bias_data.shape.size);
                    bias_in.close();
                    init(weights, bias);
                }
                Linear1D_ReLU(Env* env_ptr, dims input_dims, dims output_dims){
                    this->env = env_ptr;
                    this->input_dims = input_dims;
                    this->output_dims = output_dims;  
                    weights_data.init(env, {output_dims[1], input_dims[1]});
                    bias_data.init(env, {output_dims[1]});
                    reordered_weight_data.init_nomem(env, {output_dims[1], input_dims[1]}, true);
                }
                Linear1D_ReLU(Env* env_ptr, dims input_dims, dims output_dims, const std::vector<T>& weights, const std::vector<T>& bias){
                    this->env = env_ptr;
                    this->input_dims = input_dims;
                    this->output_dims = output_dims;  
                    weights_data.init(env, {output_dims[1], input_dims[1]});
                    bias_data.init(env, {output_dims[1]});
                    reordered_weight_data.init_nomem(env, {output_dims[1], input_dims[1]}, true);
                    init(weights, bias);
                }
                void init(const std::vector<T>& weights, const std::vector<T>& bias){
                    SetWeight(weights);
                    SetBias(bias);
                    Shape<T> shape;
                    post_ops inner_product_ops;
                    inner_product_ops.append_eltwise(
                            algorithm::eltwise_relu, 0.f, 0.f);
                    primitive_attr inner_product_attr;
                    inner_product_attr.set_post_ops(inner_product_ops);
                    memory::desc src_md = memory::desc(input_dims, shape.GetDtype(), memory::format_tag::ab);
                    memory::desc dst_md  = memory::desc(output_dims, shape.GetDtype(), memory::format_tag::ab);
                    inner_product_forward::primitive_desc linear_pd = inner_product_forward::primitive_desc(env->GetEngine(),prop_kind::forward_training, src_md, reordered_weight_data.shape.memory_desc, bias_data.shape.memory_desc, dst_md, inner_product_attr);
                    bool need_reorder_weights = linear_pd.weights_desc() != weights_data.shape.memory_desc;
                    if (need_reorder_weights) {
                        is_weight_reorder = true;
                        reordered_weight_data.mem = memory(linear_pd.weights_desc(), env->GetEngine());
                        reorder(weights_data.mem, reordered_weight_data.mem).execute(env->GetStream(), weights_data.mem, reordered_weight_data.mem);
                    }
                    
                    primitive = inner_product_forward(linear_pd);
                }
                void SetWeight(const std::vector<T>& weights){
                    this->weights_data = weights;
                }
                void SetBias(const std::vector<T>& bias){
                    this->bias_data = bias;
                }
                void forward(Tensor<T>& input, Tensor<T>& output, bool save_process = false){
                    if(first_run){
                        inner_product_args.insert({DNNL_ARG_SRC, input.mem});
                        if(is_weight_reorder){
                            inner_product_args.insert({DNNL_ARG_WEIGHTS, reordered_weight_data.mem});
                        }else{
                            inner_product_args.insert({DNNL_ARG_WEIGHTS, weights_data.mem});
                        }
                        inner_product_args.insert({DNNL_ARG_BIAS, bias_data.mem});
                        inner_product_args.insert({DNNL_ARG_DST, output.mem});
                        first_run = false;
                    }
                    primitive.execute(env->GetStream(), inner_product_args);
                    if(save_process){
                        env->GetStream().wait();
                        output.save();
                    }
                }

        };
 
    }
}}