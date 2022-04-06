# LinearList: Accelerate Pytorch ModuleList of Linear Layers via a CUDA implementation
This is a faster implementation for Pytorch ModuleList of Linear Layers using custom C++ and CUDA extensions.

## An Example
Assuming we have an input containing data from multiple sources, and we need to create a unique linear layer for each sources and apply corresponding layers to different sources. A naive implementation might be as follow:
```
import torch
import torch.nn as nn
import time

batch_size = 32
num_linears = 10000
input_dim = 7
output_dim = 7

ll = nn.ModuleList([nn.Linear(input_dim, output_dim, bias=False) for i in range(num_linears)]).to("cuda:0")
                         
def deal_list(module_list, x):
    '''
    x: input with shape [batch_size, num_linears, input_dim]
    y: output with shape [batch_size, num_linears, input_dim]
    '''
    x = x.permute(1, 0, 2).contiguous()
    outputs = []
    for i,l in enumerate(module_list):
        output = l(x[i])
        outputs.append(output)
    y = torch.stack(outputs, 1)
    return y

begin = time.time()
x = torch.ones(batch_size, num_linears, input_dim).to("cuda:0")
y = deal_list(ll, x)
end = time.time()
print(end-begin)
```
Implementing a Pytorch moduleList of linear layers in this way is straghtforward, but would induce low efficiency, especially when the number of linear layers is large. This is because the above implementation executes the computation of each layer one by one. Actually, the computations in such a moduleList of linear layers are highly parallelable. Here, I re-implement its forward and backward mechanisms in CUDA, providing a more efficient implementation.

## Install
```
cd LinearList/linearlist_cuda_extension
python setup install --user
```
If bugs like "error: invalid static_cast from type ‘const torch::OrderedDict<std::basic_string<char>, std::shared_ptr<torch::nn::Module> >’ to type ‘torch::OrderedDict<std::basic_string<char>, std::shared_ptr<torch::nn::Module> >&" are reported, please refer to https://zhuanlan.zhihu.com/p/468605263 for help.

Then you can just put the directory LinearList into your project and use it.

## Usage
```
import torch
import torch.nn as nn
from LinearList import LinearList
import time

batch_size = 32
num_linears = 10000
input_dim = 7
output_dim = 7

ll = LinearList(input_dim, output_dim, num_linears).to("cuda:0")
                         
def deal_list_faster(module_list, x):
    '''
    x: input with shape [batch_size, num_linears, input_dim]
    y: output with shape [batch_size, num_linears, input_dim]
    '''
    x_size = x.size()
    x = x.view(x_size[0], -1)
    output = module_list(x)
    y = output.view(x_size[0], x_size[1], -1)
    return y

begin = time.time()
x = torch.ones(batch_size, num_linears, input_dim).to("cuda:0")
y = deal_list_faster(ll, x)
end = time.time()
print(end-begin)
```
Implementing a list of linear layers in this way can achieve 100× acceleration in some cases. Details can be seen in comparison_of_different_implementations.ipynb.
## More Examples
graph_lstm_vae_ad_ver6 contains a implementation of [TopoMAD](https://github.com/QAZASDEDC/TopoMAD). graph_lstm_vae_ad_faster contains its faster implementation via turning all Pytorch ModuleLists of Linear Layers into LinearList implemented here. We can see the new implementation maintains the same functionality while achiving higher efficiency. 

## TODO
- Add bias support

## Reference
* Solving errors in installing pytorch cuda extensions: https://zhuanlan.zhihu.com/p/468605263
* MSRA AI-SYSTEM COURSE LABs: https://github.com/microsoft/AI-System/tree/main/Labs
* CUDA Programming model: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html 
* An Even Easier Introduction to CUDA: https://devblogs.nvidia.com/even-easier-introduction-cuda/ 
* CUSTOM C++ AND CUDA EXTENSIONS: https://pytorch.org/tutorials/advanced/cpp_extension.html
* TopoMAD: https://github.com/QAZASDEDC/TopoMAD

