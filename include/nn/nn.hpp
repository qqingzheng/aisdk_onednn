#include "nn/absrtactnn.hpp"
#include "data/tensor.hpp"
// #include "oneapi/dnnl/dnnl_types.h"
using namespace dnnl;

namespace aisdk{ namespace nn{
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
                void forward(Tensor<T>& input, Tensor<T>& output){
                    primitive.execute(env->GetEngine(),
                    {
                        {DNNL_ARG_SRC, input.mem},
                        {DNNL_ARG_DST, output.mem}
                    });
                    env->GetStream().wait();
                    output.save(output.data.data());
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
                void forward(Tensor<T>& input, Tensor<T>& output){
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
                    env->GetStream().wait();
                    output.save(output.data.data());
                }

        };
    }
}}