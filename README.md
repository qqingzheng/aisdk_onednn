# AISDK

Intel OneDNN encapsulation

# Example

```cpp
#include "aisdk.h"
#include <cstdio>

using namespace aisdk;
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
    linear.forward(input, output);
    printf("Input:");
    for(int i = 0; i < 2; ++i){
        printf("%f ", input.data[i]);
    }
    printf("\nOutput:");
    for(int i = 0; i < 4; ++i){
        printf("%f ", output.data[i]);
    }
}
int main(){
    test_Linear1D();
    return 0;
}
```

Output: 
```
Input:0.001200 0.030000 
Output:1.000061 1.000154 1.000061 1.000246
```