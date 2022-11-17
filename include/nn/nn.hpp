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
        void Concat(Env* env_ptr ,const std::vector<Tensor<T>*>& tensors, int axis, Tensor<T>& output, bool save_process = false){
            std::vector<memory::desc> src_mds;
            for(auto tensor_ptr : tensors){
                // printf("%d", ((Tensor<T>*)tensor_ptr)->mem.)
                src_mds.push_back(tensor_ptr->shape.memory_desc);
            }
            auto concat_pd = concat::primitive_desc(axis, src_mds, env_ptr->GetEngine());
            output.init(env_ptr, concat_pd.dst_desc());
            auto concat_prim = concat(concat_pd);
            std::unordered_map<int, memory> concat_args;
            for(int i = 0; i < tensors.size(); ++ i){
                concat_args.insert({DNNL_ARG_MULTIPLE_SRC + i, tensors[i]->mem});
            }
            concat_args.insert({DNNL_ARG_DST, output.mem});
            concat_prim.execute(env_ptr->GetStream(), concat_args);
            if(save_process){
                output.save();
                env_ptr->GetStream().wait();
            }
        }
        template<typename T>
        void Concat(Env* env_ptr ,const Tensor<T>& tensor1, const Tensor<T>& tensor2, int axis, Tensor<T>& output, bool save_process = false){
            std::vector<memory::desc> src_mds;
            src_mds.push_back(tensor1.shape.memory_desc);
            src_mds.push_back(tensor2.shape.memory_desc);
            auto concat_pd = concat::primitive_desc(axis, src_mds, env_ptr->GetEngine());
            auto concat_prim = concat(concat_pd);
            std::unordered_map<int, memory> concat_args;
            concat_args.insert({DNNL_ARG_MULTIPLE_SRC, tensor1.mem});
            concat_args.insert({DNNL_ARG_MULTIPLE_SRC + 1, tensor2.mem});
            concat_args.insert({DNNL_ARG_DST, output.mem});
            concat_prim.execute(env_ptr->GetStream(), concat_args);
            if(save_process){
                output.save();
                env_ptr->GetStream().wait();
            }
        }
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
                    pd = eltwise_forward::primitive_desc({
                            prop_kind::forward_inference, algorithm::eltwise_relu,
                            md.memory_desc, 
                            alpha, 
                            beta 
                        }, env->GetEngine()
                    );
                    primitive = eltwise_forward(pd);
                }
                ReLU(Env* env_ptr, const dims& dim){
                    this->env = env_ptr;
                    md = Shape<T>(env, dim);
                    pd = eltwise_forward::primitive_desc({
                            prop_kind::forward_inference, algorithm::eltwise_relu,
                            md.memory_desc, 
                            0.f, 
                            0.f 
                        }, env->GetEngine()
                    );
                    primitive = eltwise_forward(pd);
                }
                void forward(Tensor<T>& input, Tensor<T>& output, bool save_process = false){
                    primitive.execute(env->GetEngine(),
                    {
                        {DNNL_ARG_SRC, input.mem},
                        {DNNL_ARG_DST, output.mem}
                    });
                    if(save_process){
                        output.save();
                        env->GetStream().wait();
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
                    Shape<T> shape;
                    memory::desc src_md = memory::desc(input_dims, shape.GetDtype(), memory::format_tag::ab);
                    memory::desc dst_md  = memory::desc(output_dims, shape.GetDtype(), memory::format_tag::ab);
                    inner_product_forward::primitive_desc linear_pd = inner_product_forward::primitive_desc({
                            prop_kind::forward_training, src_md, reordered_weight_data.shape.memory_desc, bias_data.shape.memory_desc, dst_md}, env->GetEngine());

                    bool need_reorder_weights = linear_pd.weights_desc() != weights_data.shape.memory_desc;
                    if (need_reorder_weights) {
                        is_weight_reorder = true;
                        reordered_weight_data.mem = memory(linear_pd.weights_desc(), env->GetEngine());
                        reorder(weights_data.mem, reordered_weight_data.mem).execute(env->GetEngine(), weights_data.mem, reordered_weight_data.mem);
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
                    std::unordered_map<int, memory> inner_product_args;
                    inner_product_args.insert({DNNL_ARG_SRC, input.mem});
                    if(is_weight_reorder){
                        inner_product_args.insert({DNNL_ARG_WEIGHTS, reordered_weight_data.mem});
                    }else{
                        inner_product_args.insert({DNNL_ARG_WEIGHTS, weights_data.mem});
                    }
                    inner_product_args.insert({DNNL_ARG_BIAS, bias_data.mem});
                    inner_product_args.insert({DNNL_ARG_DST, output.mem});
                    primitive.execute(env->GetStream(), inner_product_args);
                    if(save_process){
                        output.save();
                        env->GetStream().wait();
                    }
                }

        };
        template<typename T>
        class Linear1D_ReLU : Layer<T>{
            private:
                Env* env;
                Shape<T> md;
                bool is_weight_reorder = false;
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
                            1.0f, algorithm::eltwise_relu, 0.f, 0.f);
                    primitive_attr inner_product_attr;
                    inner_product_attr.set_post_ops(inner_product_ops);
                    memory::desc src_md = memory::desc(input_dims, shape.GetDtype(), memory::format_tag::ab);
                    memory::desc dst_md  = memory::desc(output_dims, shape.GetDtype(), memory::format_tag::ab);
                    inner_product_forward::primitive_desc linear_pd = inner_product_forward::primitive_desc({
                            prop_kind::forward_training, src_md, reordered_weight_data.shape.memory_desc, bias_data.shape.memory_desc, dst_md}, inner_product_attr, env->GetEngine());
                    bool need_reorder_weights = linear_pd.weights_desc() != weights_data.shape.memory_desc;
                    if (need_reorder_weights) {
                        is_weight_reorder = true;
                        reordered_weight_data.mem = memory(linear_pd.weights_desc(), env->GetEngine());
                        reorder(weights_data.mem, reordered_weight_data.mem).execute(env->GetEngine(), weights_data.mem, reordered_weight_data.mem);
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
                    std::unordered_map<int, memory> inner_product_args;
                    inner_product_args.insert({DNNL_ARG_SRC, input.mem});
                    if(is_weight_reorder){
                        inner_product_args.insert({DNNL_ARG_WEIGHTS, reordered_weight_data.mem});
                    }else{
                        inner_product_args.insert({DNNL_ARG_WEIGHTS, weights_data.mem});
                    }
                    inner_product_args.insert({DNNL_ARG_BIAS, bias_data.mem});
                    inner_product_args.insert({DNNL_ARG_DST, output.mem});
                    primitive.execute(env->GetStream(), inner_product_args);
                    if(save_process){
                        output.save();
                        env->GetStream().wait();
                    }
                }

        };
 
    }
}}