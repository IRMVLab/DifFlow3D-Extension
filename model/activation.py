import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        # print("Try to use customized backward!!")
        x = x.clamp(min=-7, max=7) # original +-15
        return g * torch.exp(x)

# class _trunc_exp(Function):
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float32)
#     def forward(ctx, x):
#         # 1. 在 forward 内部进行 clamp
#         x_clamped = x.clamp(-15, 15)
#         # 2. 保存 clamp 后的结果给 backward 用
#         ctx.save_for_backward(x_clamped)
#         print("!!! Warning: Using trunc_exp with clamping to prevent overflow !!!")
#         return torch.exp(x_clamped)

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, g):
#         # 3. 取出 clamp 后的 x
#         x_clamped = ctx.saved_tensors[0]
#         # 4. 基于 clamp 后的 x 计算梯度，这里 exp(x_clamped) 正是 forward 的导数
#         #    不再需要重复 clamp
#         return g * torch.exp(x_clamped)

trunc_exp = _trunc_exp.apply


