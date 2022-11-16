#ifndef _ENV_H
#define _ENV_H
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_types.h"

namespace aisdk{

    enum class device_type{
        ANY = 0,
        CPU = 1,
        GPU = 2
    };

    enum class data_type{
        undef       =   dnnl_data_type_undef,
        float16     =   dnnl_f16,
        bfloat16    =   dnnl_bf16,
        float32     =   dnnl_f32,
        float64     =   dnnl_f64,
        int32       =   dnnl_s32,
        int8        =   dnnl_s8,
        uint8       =   dnnl_u8

    };

    class Env{
        private:
            device_type device;
            dnnl::engine engine;
            dnnl::stream stream;
        public:
            Env() = delete;
            Env(device_type, unsigned long);
            dnnl::engine GetEngine(){ return engine; }
            dnnl::stream GetStream(){ return stream; }
    };
};

#endif