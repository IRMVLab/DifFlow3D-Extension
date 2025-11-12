# ==============================================================================
# Copyright (c) 2024 Zehan Zheng. All Rights Reserved.
# LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis
# CVPR 2024
# https://github.com/ispc-lab/LiDAR4D
# Apache License 2.0
# ==============================================================================

import torch
import torch.nn as nn
import math
import numpy as np
import tinycudann as tcnn


class FreqEmbed(nn.Module):
    def __init__(
            self,
            num_freqs, 
            linspace=True,
        ):
        super().__init__()
        self.embed_functions = [torch.sin, torch.cos]
        if linspace:
            self.freqs = torch.linspace(1, num_freqs + 1, steps=num_freqs)
        else:
            exps = torch.linspace(0, num_freqs - 1, steps=num_freqs)
            self.freqs = 2**exps

    def forward(self, x):
        embed = []
        for f in self.embed_functions:
            for freq in self.freqs:
                embed.append(f(freq * x * torch.pi))
        embed = torch.cat(embed, -1)

        return embed


class FlowField(nn.Module):
    def __init__(
            self,
            input_dim=4,
            num_layers=3,
            hidden_dim=64,
            use_freq=False,
            num_freqs=6,
            use_grid=True,
            num_basis=4,
            n_levels=8,
            n_features_per_level=8,
            base_resolution=32,
            max_resolution=8192,
            log2_hashmap_size=18,
        ):
        super().__init__()
        self.use_freq = use_freq
        self.use_grid = use_grid
        self.input_dim = 0

        if use_freq:
            self.freq_enc = FreqEmbed(num_freqs=num_freqs)
            self.input_dim += input_dim * num_freqs * 2

        if use_grid:
            per_level_scale = np.exp2(np.log2(max_resolution / base_resolution) / (n_levels - 1))
            self.grid_enc = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": n_levels,
                    "n_features_per_level": n_features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_resolution,
                    "per_level_scale": per_level_scale,
                },
            )
            self.n_levels = n_levels
            self.n_features_per_level = n_features_per_level
            self.num_basis = num_basis
            self.input_dim += self.grid_enc.n_output_dims // self.num_basis

        # Hashgrid encoder allows for smaller MLP
        layer_list = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 6 # forward & backward flow
            else:
                out_dim = hidden_dim
            layer_list.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != num_layers - 1:
                layer_list.append(nn.ReLU())
        self.mlp = nn.Sequential(*layer_list)

        torch.nn.init.normal_(self.mlp[-1].weight.data, 0, 0.001)

    def interpT(self, x, t):
        # temporal interpolation in feature dimension
        x = x.view(-1, self.n_levels, self.n_features_per_level)
        num_basis = self.num_basis
        x = torch.chunk(x, num_basis, dim=-1)
        T = [i/(num_basis-1) for i in range(num_basis)]
        basis = lambda j: [(t - T[m]) / (T[j] - T[m]) for m in range(num_basis) if m != j]
        x = sum([math.prod(basis(i)) * x[i] for i in range(num_basis)])
        x = x.view(-1, self.input_dim)
        return x
    # def interpT(self, x, t):
    #     # temporal interpolation in feature dimension, try to avoid numpy and math.prod, but still cause nan or inf in fp16 training
    #     x = x.view(-1, self.n_levels, self.n_features_per_level)
    #     num_basis = self.num_basis
        
    #     # Ensure t is a scalar on the correct device
    #     t_scalar = t.item()
        
    #     # Prepare basis points T in PyTorch
    #     T = torch.linspace(0.0, 1.0, steps=num_basis, device=x.device)

    #     # Use PyTorch broadcasting to calculate basis functions
    #     # This is much more stable and keeps everything on the GPU
    #     weights = []
    #     for j in range(num_basis):
    #         # Numerator: (t - T[m]) for all m != j
    #         # Denominator: (T[j] - T[m]) for all m != j
    #         terms = [(t_scalar - T[m]) / (T[j] - T[m]) for m in range(num_basis) if m != j]
    #         weights.append(torch.prod(torch.stack(terms)))
            
    #     weights = torch.stack(weights) # Shape: [num_basis]

    #     # Chunk the feature tensor
    #     x_chunks = torch.chunk(x, num_basis, dim=-1) # List of num_basis tensors
    #     x_stacked = torch.stack(x_chunks, dim=-1) # Shape: [N, n_levels, features_per_chunk, num_basis]
        
    #     # Weighted sum using einsum for clarity and efficiency
    #     # w[b] * x[..., c, b] -> out[..., c]
    #     x = torch.einsum('b, ...cb -> ...c', weights, x_stacked)
        
    #     x = x.reshape(-1, self.input_dim) # Use reshape which is more flexible
    #     return x

    def forward(self, xt):
        ## xt.shape: [N, 4], in [0, 1] 
        h = []
        if self.use_freq:
            x = self.freq_enc(xt)
            h.append(x)

        if self.use_grid:
            x = xt[:, :3]
            t = xt[0, 3]
            # --- 关键诊断点 ---
            # grid_features = self.grid_enc(x).float()
            
            # 立即检查 grid_features
            # print("++++++++++++++++++++++++++++")
            # print(f"Stats for grid_features (output of grid_enc):")
            # print(f"  min={grid_features.min().item()}, max={grid_features.max().item()}, mean={grid_features.mean().item()}")
            # if torch.isnan(grid_features).any():
            #     print("!!!!!! FATAL: NaN found in grid_features IMMEDIATELY after grid_enc !!!!!!")
            # if torch.isinf(grid_features).any():
            #     print("!!!!!! FATAL: Inf found in grid_features IMMEDIATELY after grid_enc !!!!!!")
            # print("++++++++++++++++++++++++++++")
            # --- 诊断结束 ---, 正常 没有 NaN 或 Inf



            x = self.grid_enc(x).float()
            with torch.cuda.amp.autocast(enabled=False):x = self.interpT(x.float(), t.float())
            h.append(x)

        h = torch.cat(h, dim=-1)
        # print("++++++++++++++++++++++++++++", f"Input to MLP (h) stats: min={h.min()}, max={h.max()}, mean={h.mean()}")
        if torch.isnan(h).any() or torch.isinf(h).any():
            print("!!! Warning: NaN or Inf found in input to MLP !!!")
        flow = self.mlp(h) ##### lead to overflow 

        return flow



if __name__ == '__main__':
    encoder = FlowField().cuda()
    x = torch.rand(100, 4).cuda()
    flow = encoder(x)
    print(flow.shape)