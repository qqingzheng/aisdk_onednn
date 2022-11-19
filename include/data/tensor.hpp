#ifndef _TENSOR_H
#define _TENSOR_H

#include "oneapi/dnnl/dnnl.hpp"
#include "env/env.h"
#include <typeinfo>


namespace aisdk{

    typedef std::vector<long> dims;
    typedef std::vector<dims> dims_list;
    typedef void* data_vector;

    template<typename T>
    class Shape{
        private: 
        public:
            unsigned int size;
            dnnl::memory::desc memory_desc;
            Shape(){}
            
            Shape(Env* env, const dims& dims, bool memory_format_any=false){
                size = GetTotalLengthOfDims(dims);
                if(dims.size() > 12) throw std::runtime_error("dims range 1-12");
                auto dtype = GetDtype();
                if(!memory_format_any){
                    memory_desc = dnnl::memory::desc(dims, dtype, (dnnl::memory::format_tag)(dims.size() + 1));
                }else{
                    memory_desc = dnnl::memory::desc(dims, dtype, dnnl::memory::format_tag::any);
                }
            }
            Shape(Env* env, dnnl::memory::desc& memory_desc){
                // memory_desc.get_size()
                this->memory_desc = memory_desc;
            }
            dnnl::memory::data_type GetDtype(){
                if(typeid(T) == typeid(int32_t)){
                    return dnnl::memory::data_type::s32;
                }else if(typeid(T) == typeid(float)){
                    return dnnl::memory::data_type::f32;
                }else if(typeid(T) == typeid(double)){
                    return dnnl::memory::data_type::f64;
                }else{
                    throw std::runtime_error("this type is not supported");
                }
            }
            inline int GetTotalLengthOfDims(const dims& dims){
                int length = 1;
                for(auto dim : dims) length *= dim;
                return length;
            }
    };
    
    template<typename T>
    class Tensor{
        private:
            Env* env;
            inline void SetCommonDesc(Env* env, const dims& dims, bool memory_format_any){
                this->env = env;
                this->shape = Shape<T>(env, dims, memory_format_any); 
            }
            // inline void WriteDataIntoMem(const std::vector<T>& data) { dnnl::memory::write_to_dnnl_memory(data.data(), this->mem); }
        public:
            dnnl::memory mem;
            std::vector<T> data;
            Shape<T> shape;
            Tensor(){  }
            Tensor(Env* env, const Tensor<T>& src, const dims& dims, int offset){ 
                SetCommonDesc(env, dims, false);
                if(shape.size > src.shape.size - offset) throw std::runtime_error("too long");
                CreateMem();
                this->data.resize(this->shape.size);
                mem.set_data_handle((T*)src.mem.get_data_handle() + offset);
            }
            Tensor(Env* env, const Tensor<T>& src, const dims& dims, int offset, const std::vector<T>& data){ 
                SetCommonDesc(env, dims, false);
                if(shape.size > src.shape.size - offset) throw std::runtime_error("too long");
                CreateMem();
                mem.set_data_handle((T*)src.mem.get_data_handle() + offset);
                this->data.resize(this->shape.size);
                this->data = data; 
                load(this->data.data());
            }
            Tensor(Env* env, const dims& dims, const std::vector<T>& data, bool memory_format_any=false){ init(env, dims, data, memory_format_any); }
            Tensor(Env* env, const dims& dims, bool memory_format_any=false){ init(env, dims, memory_format_any); }
            Tensor(Env* env, dnnl::memory::desc memory_desc){ init(env, memory_desc); }
            void CreateMem(){  mem = dnnl::memory(this->shape.memory_desc, env->GetEngine()); }
            void init(Env* env, const dims& dims, const std::vector<T>& data, bool memory_format_any=false){
                SetCommonDesc(env, dims, memory_format_any);
                CreateMem();
                this->data.resize(this->shape.size);
                this->data = data; 
                load(this->data.data());
            }
            void init(Env* env, const dims& dims, bool memory_format_any=false){
                SetCommonDesc(env, dims, memory_format_any);
                CreateMem();
                this->data.resize(this->shape.size);
            }
            void init_nomem(Env* env, const dims& dims, bool memory_format_any=false){
                SetCommonDesc(env, dims, memory_format_any);
                this->data.resize(this->shape.size);
            }
            void offset(int offset){
                mem.set_data_handle((void*)((T*)mem.get_data_handle() - offset));
            }
            void init(Env* env, dnnl::memory::desc memory_desc){
                this->shape.memory_desc = memory_desc;
                this->shape.size = this->shape.GetTotalLengthOfDims(this->shape.memory_desc.dims());
                CreateMem();
                data.resize(this->shape.size);
            }
            void operator=(const std::vector<T>& data){
                this->data = data; 
                load(this->data.data());
            }
            void operator=(const Tensor<T>& data){
                this->data = data.data; 
                load(this->data.data());
            }
            dnnl::memory& GetMem(){ return mem; }
            void load(void *);
            void save(void *);
            void save();
    };
    template<typename T>
    void Tensor<T>::save(){
        save(this->data.data());
    }
    template<typename T>
    void Tensor<T>::save(void* handle){
        dnnl::engine eng = mem.get_engine();
        size_t size = mem.get_desc().get_size();
        if (!handle) throw std::runtime_error("handle is nullptr.");

#ifdef DNNL_WITH_SYCL
        bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
                && eng.get_kind() == dnnl::engine::kind::cpu);
        bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
                && eng.get_kind() == dnnl::engine::kind::gpu);
        if (is_cpu_sycl || is_gpu_sycl) {
            auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
            if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
                auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
                auto src = buffer.get_access<::sycl::access::mode::read>();
                uint8_t *src_ptr = src.get_pointer();
                if (!src_ptr)
                    throw std::runtime_error("get_pointer returned nullptr.");
                for (size_t i = 0; i < size; ++i)
                    ((uint8_t *)handle)[i] = src_ptr[i];
            } else {
                assert(mkind == dnnl::sycl_interop::memory_kind::usm);
                uint8_t *src_ptr = (uint8_t *)mem.get_data_handle();
                if (!src_ptr)
                    throw std::runtime_error("get_data_handle returned nullptr.");
                if (is_cpu_sycl) {
                    for (size_t i = 0; i < size; ++i)
                        ((uint8_t *)handle)[i] = src_ptr[i];
                } else {
                    auto sycl_queue
                            = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                    sycl_queue.memcpy(handle, src_ptr, size).wait();
                }
            }
            return;
        }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if (eng.get_kind() == dnnl::engine::kind::gpu) {
            void *mapped_ptr = mem.map_data();
            if (mapped_ptr) std::memcpy(handle, mapped_ptr, size);
            mem.unmap_data(mapped_ptr);
            return;
        }
#endif
        if (eng.get_kind() == dnnl::engine::kind::cpu) {
            uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
            if (!src) throw std::runtime_error("get_data_handle returned nullptr.");
            for (size_t i = 0; i < size; ++i)
                ((uint8_t *)handle)[i] = src[i];
            return;
        }
        throw std::runtime_error("save error!");
    }

    template<typename T>
    void Tensor<T>::load(void* handle){
        dnnl::engine eng = mem.get_engine();
        size_t size = mem.get_desc().get_size();
        if (!handle) throw std::runtime_error("handle is nullptr.");
#ifdef DNNL_WITH_SYCL
        bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
                && eng.get_kind() == dnnl::engine::kind::cpu);
        bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
                && eng.get_kind() == dnnl::engine::kind::gpu);
        if (is_cpu_sycl || is_gpu_sycl) {
            auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
            if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
                auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
                auto dst = buffer.get_access<::sycl::access::mode::write>();
                uint8_t *dst_ptr = dst.get_pointer();
                if (!dst_ptr)
                    throw std::runtime_error("get_pointer returned nullptr.");
                for (size_t i = 0; i < size; ++i)
                    dst_ptr[i] = ((uint8_t *)handle)[i];
            } else {
                assert(mkind == dnnl::sycl_interop::memory_kind::usm);
                uint8_t *dst_ptr = (uint8_t *)mem.get_data_handle();
                if (!dst_ptr)
                    throw std::runtime_error("get_data_handle returned nullptr.");
                if (is_cpu_sycl) {
                    for (size_t i = 0; i < size; ++i)
                        dst_ptr[i] = ((uint8_t *)handle)[i];
                } else {
                    auto sycl_queue
                            = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                    sycl_queue.memcpy(dst_ptr, handle, size).wait();
                }
            }
            return;
        }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if (eng.get_kind() == dnnl::engine::kind::gpu) {
            void *mapped_ptr = mem.map_data();
            if (mapped_ptr) std::memcpy(mapped_ptr, handle, size);
            mem.unmap_data(mapped_ptr);
            return;
        }
#endif

        if (eng.get_kind() == dnnl::engine::kind::cpu) {
            uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
            if (!dst) throw std::runtime_error("get_data_handle returned nullptr.");
            for (size_t i = 0; i < size; ++i)
                dst[i] = ((uint8_t *)handle)[i];
            return;
        }
        throw std::runtime_error("load error!");

    }

};

#endif