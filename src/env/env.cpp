#include "env.h"
using namespace aisdk;
Env::Env(device_type device, unsigned long idx){
    this->engine = dnnl::engine((dnnl::engine::kind)device, idx);
    this->stream = dnnl::stream(this->engine);
}