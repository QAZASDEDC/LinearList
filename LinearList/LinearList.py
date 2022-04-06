import torch
import torch.nn as nn
import linearlist_cuda

class LinearListFunction(torch.autograd.Function):
    num_linears = None
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        #output = input.mm(weight.t())
        #output = mylinear_cpp.forward(input, weight)
        output = linearlist_cuda.forward(input, weight, LinearListFunction.num_linears)

        #weight_list = weight.view(weight.size(0), LinearListFunction.num_linears, -1).permute(1, 0, 2).contiguous()
        #input_list = input.view(input.size(0), LinearListFunction.num_linears, -1).permute(1, 0, 2).contiguous()
        #right_output = []
        #for i in range(LinearListFunction.num_linears):
        #    right_output.append(torch.mm(input_list[i],  weight_list[i].t()))
        #right_output = torch.cat(right_output, 1)
        #print("FORWARD: right_output==output?")
        #print(right_output-output[0])

        #return right_output
        return output[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        #grad_input = grad_weight = None
        #grad_input = grad_output.mm(weight)
        #grad_weight = grad_output.t().mm(input)
        #grad_input, grad_weight = mylinear_cpp.backward(grad_output, input, weight)
        num_linears = grad_output.size(-1)//weight.size(0)

        if (not grad_output.is_contiguous()):
            print(grad_output)
            grad_output = grad_output.contiguous()

        #print("grad_output:", grad_output, grad_output.shape)
        #print("input:", input, input.shape)
        grad_input, grad_weight = linearlist_cuda.backward(grad_output, input, weight, num_linears)

        #print("grad_input:", grad_input, grad_input.shape)
        #print("grad_weight:", grad_weight, grad_weight.shape)


        #grad_output_list = grad_output.view(grad_output.size(0), num_linears, -1).permute(1, 0, 2).contiguous()
        #weight_list = weight.view(weight.size(0), num_linears, -1).permute(1, 0, 2).contiguous()
        #input_list = input.view(input.size(0), num_linears, -1).permute(1, 0, 2).contiguous()
        #
        #right_grad_input = []
        #for i in range(num_linears):
        #    right_grad_input.append(torch.mm(grad_output_list[i], weight_list[i]))
        #right_grad_input = torch.cat(right_grad_input, 1)
        #print("right_grad_input", right_grad_input, right_grad_input.shape)

        #right_grad_weight = []
        #for i in range(num_linears):
        #    right_grad_weight.append(torch.mm(grad_output_list[i].t(), input_list[i]))
        #right_grad_weight = torch.cat(right_grad_weight, 1)
        #print("right_grad_weight", right_grad_weight, right_grad_weight.shape)

        #print("right_grad_input==grad_input?")
        #print(right_grad_input-grad_input)

        #print("right_grad_weight==grad_weight?")
        #print(right_grad_weight-grad_weight)

        #print("over-----------------------------------------------------")

        #return right_grad_input, right_grad_weight
        return grad_input, grad_weight

class LinearList(nn.Module):
    def __init__(self, per_input_features, per_output_features, num_linears):
        super(LinearList, self).__init__()
        self.per_input_features = per_input_features
        self.per_output_features = per_output_features
        self.num_linears = num_linears
        self.weight = nn.Parameter(torch.Tensor(per_output_features, per_input_features*num_linears))
        self.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, input):
        LinearListFunction.num_linears = self.num_linears
        return LinearListFunction.apply(input, self.weight)