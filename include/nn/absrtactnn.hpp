#include "data/tensor.hpp"

namespace aisdk{ namespace nn{
    template<typename T>
    class Layer{
        virtual void forward(Tensor<T>& input, Tensor<T>& output, bool save_process = false) = 0;
    };

}}

