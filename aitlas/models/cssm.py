"""CSSM: Efficient Remote Sensing Change Detection with Change State Space Models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
import math
import warnings

from ..base import BaseChangeDetection

# -----------------------------------------------------------------------------
# Mamba / SSM Components (from method/MambaCSSM.py)
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class MambaBlock_CD(nn.Module):
    def __init__(self, d_model, d_conv, d_state, bias=True, conv_bias=True):
        """
        A Mamba block adapted for Change Detection (processes two inputs).
        Based on CSSM-main/method/MambaCSSM.py
        """
        super().__init__()
        
        self.norm = RMSNorm(d_model=d_model)

        self.d_inner = 2 * d_model
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D_p = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, t1, t2):
        # Preservation of residual connection
        ee1 = t1 
        ee2 = t2
        
        # Process t1
        (b, l, d) = t1.shape
        t1 = self.norm(t1)
        
        t1_and_res = self.in_proj(t1)  # shape (b, l, 2 * d_in)
        (t1, res1) = t1_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        t1 = rearrange(t1, 'b l d_in -> b d_in l')
        t1 = self.conv1d(t1)[:, :, :l]
        t1 = rearrange(t1, 'b d_in l -> b l d_in')
        
        t1 = F.silu(t1)

        # Process t2
        (b, l, d) = t2.shape
        t2 = self.norm(t2)
        
        t2_and_res = self.in_proj(t2)  # shape (b, l, 2 * d_in)
        (t2, res2) = t2_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        t2 = rearrange(t2, 'b l d_in -> b d_in l')
        t2 = self.conv1d(t2)[:, :, :l]
        t2 = rearrange(t2, 'b d_in l -> b l d_in')
        
        t2 = F.silu(t2)

        # CSSM interaction
        y1, y2 = self.cssm(t1, t2)
        
        y1 = y1 * F.silu(res1)
        y2 = y2 * F.silu(res2)
        
        output1 = self.out_proj(y1)
        output2 = self.out_proj(y2)

        return output1 + ee1, output2 + ee2

    def cssm(self, t1, t2):
        (d_in, n) = self.A_log.shape
        
        # t1 branch parameters
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        t1_dbl = self.x_proj(t1)  # (b, l, dt_rank + 2*n)
        (delta, B, C) = t1_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        # t2 branch parameters
        A_prim = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D_prim = self.D_p.float()


        t2_dbl = self.x_proj(t2)  # (b, l, dt_rank + 2*n)
        (delta_prim, B_prim, C_prim) = t2_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta_prim: (b, l, dt_rank). B, C: (b, l, n)        
        delta_prim = F.softplus(self.dt_proj(delta_prim))  # (b, l, d_in)

        y = self.selective_scan(t1, t2, delta, delta_prim, A, B, C, D, A_prim, B_prim, C_prim, D_prim)
        return y
    
    def selective_scan(self, t1, t2, delta, delta_prim, A, B, C, D, A_prim, B_prim, C_prim, D_prim):
        """
        Perform the discrete selective scan.
        
        NOTE: Deviation from original source code to align with the CSSM Paper mathematically.
        The original paper defines the state update for the CSSM block as:
            h_t = \\bar{A} h_{t-1} + || \\bar{B}'_t z_{post,t} - \\bar{B}_t z_{pre,t} ||_1
        
        In State Space Models, the discretization of the B matrices depends on the step-size \\Delta.
        Therefore, \\bar{B}_t must be discretized using \\Delta_{pre} (delta), and \\bar{B}'_t 
        must be discretized using \\Delta_{post} (delta_p). 
        
        The original implementation accidentally overwrote the pre-change delta with the 
        post-change delta due to a variable naming collision. We correct this here by passing 
        and using both distinct deltas to preserve the intended mathematical formulation.
        """
        (b, l, d_in) = t1.shape
        n = A.shape[1]

        # Discretize using respective deltas
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, t1, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        deltaA_prim = torch.exp(einsum(delta_prim, A_prim, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u_prim = einsum(delta_prim, B_prim, t2, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # --- Scan 1 (Pre-Change representation) ---
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            # The core CSSM interaction: State updates with absolute difference of discretized inputs
            x = deltaA[:, i] * x + torch.abs(deltaB_u[:, i] - deltaB_u_prim[:,i])
            y1 = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y1)
        y1 = torch.stack(ys, dim=1)
        y1 = y1 + t1 * D

        # --- Scan 2 (Post-Change representation) ---
        x = torch.zeros((b, d_in, n), device=deltaA_prim.device)
        ys = []    
        for i in range(l):
            x = deltaA_prim[:, i] * x + torch.abs(deltaB_u[:, i] - deltaB_u_prim[:,i])
            y2 = einsum(x, C_prim[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y2)
        y2 = torch.stack(ys, dim=1)
        y2 = y2 + t2 * D_prim
    
        return y1, y2

class MambaCSSM(nn.Module):
    def __init__(self, num_layers, d_model, d_conv, d_state, bias=True, conv_bias=True):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock_CD(d_model, d_conv, d_state, bias=bias, conv_bias=conv_bias) 
            for _ in range(num_layers)
        ])

    def forward(self, t1, t2):
        for layer in self.layers:
            t1, t2 = layer(t1, t2)
        return t1, t2 

# -----------------------------------------------------------------------------
# Main Model Architecture (from method/Model.py)
# -----------------------------------------------------------------------------

class CSSMModel(nn.Module):
    """
    Implementation of CSSM: Efficient Remote Sensing Change Detection with Change State Space Models
    Based on: https://github.com/Elman295/CSSM
    Original paper: https://arxiv.org/abs/2504.11080
    DOI: 10.48550/arXiv.2504.11080
    """
    def __init__(self, in_channels=3, output_classes=2, img_size=256):
        super(CSSMModel, self).__init__()

        # --- Encoder Conv ---
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, 16, 3, 1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.mp_block_1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.mp_block_2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.mp_block_3 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.mp_block_4 = nn.MaxPool2d(2, 2, return_indices=True)

        # --- Mamba Bottleneck ---
        # The original code used d_model=256 which corresponds to the flattened spatial resolution
        # of a 16x16 feature map (from a 256x256 input).
        # To maintain architectural integrity while allowing resize, we calculate d_model dynamically.
        # 4 maxpools = factor of 16.
        feature_size = img_size // 16
        mamba_d_model = feature_size * feature_size 
        
        self.mamba = MambaCSSM(num_layers=4, d_model=mamba_d_model, d_conv=4, d_state=16)

        # --- Decoder Deconv ---
        self.mpu_block_4 = nn.MaxUnpool2d(2, 2)
        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.ReLU()
        )
        self.deconv_4_block = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, 1, padding=1),
            nn.ReLU()
        )

        self.mpu_block_3 = nn.MaxUnpool2d(2, 2)
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.ReLU()
        )
        self.deconv_3_block = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, 1, padding=1),
            nn.ReLU()
        )

        self.mpu_block_2 = nn.MaxUnpool2d(2, 2)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU()
        )
        self.deconv_2_block = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, 1, padding=1),
            nn.ReLU()
        )

        self.mpu_block_1 = nn.MaxUnpool2d(2, 2)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, padding=1),
            nn.ReLU()
        )
        self.deconv_1_block = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 6, 3, 1, padding=1),
            nn.ReLU()
        )

        self.conv_final = nn.Conv2d(6, output_classes, 1, 1)

    def forward(self, t1, t2):
        # Concatenate inputs (Early Fusion)
        t = torch.cat([t1, t2], dim=1)

        # Encoder
        x1 = self.conv_block_1(t)
        f1, i1 = self.mp_block_1(x1)
        
        x2 = self.conv_block_2(f1)
        f2, i2 = self.mp_block_2(x2)
        
        x3 = self.conv_block_3(f2)
        f3, i3 = self.mp_block_3(x3)
        
        x4 = self.conv_block_4(f3)
        f4, i4 = self.mp_block_4(x4)

        # Bottleneck Preparation
        b, c, h, w = f4.shape
        f4_t1 = f4[:, :c//2, :, :]
        f4_t2 = f4[:, c//2:, :, :]

        # Transformation for Mamba: 
        # Original: f4_t1.view((-1, 64, 16*16)) -> (Batch, Channels, Spatial_Pixels)
        # Note: CSSM treats Channels (64) as Sequence Length, and Spatial (16*16=256) as Embedding Dim.
        f4_t1 = f4_t1.reshape(b, 64, h*w) 
        f4_t2 = f4_t2.reshape(b, 64, h*w) 
        
        # Mamba Processing
        f5_t1, f5_t2 = self.mamba(f4_t1, f4_t2)
        
        # Restore Shapes
        f5_t1 = f5_t1.reshape(b, 64, h, w)
        f5_t2 = f5_t2.reshape(b, 64, h, w)

        f5 = torch.cat([f5_t1, f5_t2], dim=1)

        # Decoder
        f6 = self.mpu_block_4(f5, i4)
        f7 = self.conv_4(torch.cat((x4, f6), dim=1))
        f8 = self.deconv_4_block(f7)

        f9 = self.mpu_block_3(f8, i3, output_size=x3.size())
        f10 = self.conv_3(torch.cat((f9, x3), dim=1))
        f11 = self.deconv_3_block(f10)

        f12 = self.mpu_block_2(f11, i2)
        f13 = self.conv_2(torch.cat((f12, x2), dim=1))
        f14 = self.deconv_2_block(f13)

        f15 = self.mpu_block_1(f14, i1)
        f16 = self.conv_1(torch.cat((f15, x1), dim=1))
        f17 = self.deconv_1_block(f16)
        
        f18 = self.conv_final(f17)

        return f18.contiguous()

# -----------------------------------------------------------------------------
# Aitlas Wrapper
# -----------------------------------------------------------------------------

class CSSM(BaseChangeDetection):
    """
    Wrapper for CSSM
    """
    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            warnings.warn(
                "Pretrained weights are not available for this model. "
                "Model will be initialized with random weights."
            )
        
        self.model = CSSMModel(
            in_channels=3, #self.config.in_channels,
            output_classes=self.config.num_classes,
            img_size=256 #self.config.img_size
        )

    def forward(self, x1, x2):
        return self.model(x1, x2)