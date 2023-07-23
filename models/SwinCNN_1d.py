# -----------------------------------------------------------
# 1D Swin Transformer for sequence training
# Copyright (c) 2022 Chenguang Wan
# Licensed under The Apache License [see LICENSE for details]
# Written by Chenguang Wan
# -----------------------------------------------------------

from typing import Optional, Callable, List, Any

import torch
import torchvision.ops
from torch import nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torch import nn, Tensor
import torch.fx


class Permute(nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """
    def __init__(self, dims) -> None:
        super().__init__()
        self.dims = dims
    
    def forward(self, x:Tensor) -> Tensor:
        return torch.permute(x, self.dims)
        
class MLP(nn.Module):
    """The MLP for SwinTrans 1d block. Details can be found at https://arxiv.org/abs/2207.05695
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, activation_layer=nn.GELU, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = activation_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class PatchMerging(nn.Module):
    """
    Args:
        dim (int): the channels (default last dimension) of input tensor.
        norm_layer (Callable): layer normarlization function.
    Returns:
        Tensor with layout of [..., L/2, 2*C]
    """
    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., L, C]
        Returns:
            Tensor with layout of [..., L/2, 2*C]
        """
        L, _ = x.shape[-2:]
        # let x be divisible by 4
        x = F.pad(x, (0,0,0, 4 - L % 4))
        
        # set step to `4`
        x0 = x[..., 0::4, :]
        x1 = x[..., 1::4, :]
        x2 = x[..., 2::4, :]
        x3 = x[..., 3::4, :]
        
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class ChannelChange(torch.nn.Sequential):
    def __init__(
        self,
        input_dim, 
        embed_dim,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        layers: List[nn.Module] = nn.ModuleList()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        layers = nn.Sequential(
                # Make Channel last to channel first
                Permute([0, 2, 1]),
                # Conv1d input: [B, C_{in}, L_{in}]
                # Conv1d output: [B, C_{out}, L_{out}]
                # modify the channel of input.
                nn.Conv1d(input_dim, embed_dim, kernel_size=1, stride=1),
                # Make channel last to channel first
                Permute([0, 2, 1]),    
                norm_layer(embed_dim),
            )
        super().__init__(*layers)


# @torch.fx.wrap
def shifted_window_attention_1d(
    inputs: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_postion_bias: Tensor,
    window_size: int,
    num_heads: int,
    shift_size: int,
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    ):
    # inputs [B, L, C]
    B, L, C = inputs.shape
    # pad L;
    pad_L = (window_size - L % window_size) % window_size
    x = F.pad(inputs, (0, 0, 0, pad_L))
    _, pad_L, _ = x.shape
    
    # If window size is larger than feature size, there is no need to shift window
    if window_size >= pad_L:
        shift_size = 0
    
    # cyclic shift
    if shift_size> 0:
        x = torch.roll(x, -shift_size, dims=(1))
    
    # partition windows
    num_windows = (pad_L // window_size)
    # x is reshaped
    x = x.view(B, pad_L // window_size, window_size, C).reshape(B*num_windows, window_size, C)
    
    # multi-head attention
    # qkv: [B*num_windows, window_size, C * 3]
    qkv = F.linear(x, qkv_weight, qkv_bias)
    # 3，x.size(0)|[B*num_windows], num_heads, x.size(1)|[window_size], C // num_heads
    # 3 means q k v 
    # number head should factor of C.
    # X size: [B, pad_L, C]
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    # `$q/\sqrt{d}$`
    q = q * (C // num_heads) ** -0.5
    # transpose the last two dimension.
    attn = q.matmul(k.transpose(-2, -1))
    attn = attn + relative_postion_bias
    if shift_size > 0:
        # generate attention mask
        attn_mask = x.new_zeros(pad_L)
        l_slices = ((0, -window_size), (-window_size, -shift_size), (-shift_size, None))
        count = 0
        for l in l_slices:
            attn_mask[l[0]:l[1]] = count
            count += 1
        attn_mask = attn_mask.view(pad_L // window_size, window_size)
        attn_mask = attn_mask.reshape(num_windows, window_size)
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        # float(-100) is a big negative value for attn
        # 0: -window_size fill to 0. others -100
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout)
    
    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout)
    
    # reverse windows
    x = x.view(B, pad_L // window_size, window_size, C)
    x = x.reshape(B, pad_L, C)
    
    # reverse cyclic shift
    if shift_size > 0:
        x = torch.roll(x, shift_size, dims=(1))
    
    # unpad featrues
    x = x[:, :L, :].contiguous()
    return x

class ShiftedWindowAttention(nn.Module):
    """
    encapsulation for func `shifted_window_attention`
    """
    # The window size is related to patch.
    def __init__(
        self,
        dim: int,
        window_size: int,
        shift_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,   
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        
        # shared Weights
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        
        # The maximum move length is `2*window_size - 1`
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size - 1), num_heads)
        )
        coords_l = torch.arange(self.window_size)
        # [window_size,  window_size]  
        relative_coords = coords_l[:, None] - coords_l[None, :] 
        # For the relative index selection.
        relative_coords += self.window_size -1
        # This step for 1d is different from 2d.
        # 2d requires sum or some other actions. But 1d donot need it.
        # [window_size*window_size]
        relative_position_index = relative_coords.view(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        # an init method.
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Tensor with layout of [B, L, C]
        Returns:
            Tensor with same layout as input, i.e. [B, L, C]
        """        
        N = self.window_size
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]
        # window_size num_window
        relative_position_bias = relative_position_bias.view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        return shifted_window_attention_1d(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
        )

class SwinTransformerBlock_1d(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0, # dropout for attention projection.
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth_prob = stochastic_depth_prob
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), activation_layer=nn.GELU,  dropout=dropout)   

        # initialization built-in __init__     
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = x + torchvision.ops.stochastic_depth(self.attn(self.norm1(x)), self.stochastic_depth_prob, "row", self.training)
        x = x + torchvision.ops.stochastic_depth(self.mlp(self.norm2(x)), self.stochastic_depth_prob, "row", self.training)
        return x

class Swin1dClass(nn.Module):
    """  Swin Transformer for seq to classification.
    """
    def __init__(
        self,
        patch_size: int,
        input_dim: int,
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if block is None:
            block = SwinTransformerBlock_1d
        
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        self.use_checkpoint = use_checkpoint
        layers: List[nn.Module] = []
        # split sequence to non-overlapping patches
        layers.append(
            nn.Sequential(
                # Make Channel last to channel first
                Permute([0, 2, 1]),
                # Conv1d input: [B, C_{in}, L_{in}]
                # Conv1d output: [B, C_{out}, L_{out}]
                nn.Conv1d(input_dim, embed_dim, kernel_size=patch_size, stride=patch_size),
                # Make channel last to channel first
                Permute([0, 2, 1]),    
                norm_layer(embed_dim),
            )
        )
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2 ** i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size= [0 if i_layer % 2 == 0 else window_size // 2][0],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(PatchMerging(dim, norm_layer))
        self.features = layers
        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.features:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        x = self.norm(x)
        x = x.permute(2,0,1)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = x.permute(1, 0)
        x = self.dense(x)
        return x        

class Swin1dSeq(nn.Module):
    """ Swin Transformer for sequence for keeping causality. The Model can be used on seq2seq. 
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        output_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None, 
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if block is None:
            block = SwinTransformerBlock_1d
        
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        self.use_checkpoint = use_checkpoint
        
        layers: List[nn.Module] = nn.ModuleList()
        # split sequence to non-overlapping patches
        layers.append(
            ChannelChange(input_dim, embed_dim)
        )
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage = nn.ModuleList()
            dim = embed_dim * 2 ** i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size= [0 if i_layer % 2 == 0 else window_size // 2][0],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # # add patch merging layer
            if i_stage < (len(depths) - 1):
                # after ChanelChange layer the output_dim (Channel) will be dim * 2.
                layers.append(ChannelChange(dim, dim * 2))
        self.features = layers    
        num_channels = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_channels)
        self.dense = nn.Linear(num_channels, output_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.features:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        x = self.norm(x)
        x = self.dense(x)
        return x        

# multiple layers 1D Swin sequence
class Swin1dSeqMultipleLayer(nn.Module):
    """  A multiple layer implenmentation of Swin1dSeq
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        output_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        num_layers = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None, 
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.models = nn.ModuleList()
        for _ in range(num_layers-1):
            self.models.append(
                Swin1dSeq(input_dim, 
                          embed_dim, 
                          input_dim, 
                          depths, 
                          num_heads, 
                          window_size,
                          mlp_ratio,
                          dropout,
                          attention_dropout,
                          stochastic_depth_prob,
                          norm_layer,
                          block,
                          use_checkpoint))
            
        self.models.append(Swin1dSeq(input_dim, 
                                embed_dim, 
                                output_dim, 
                                depths, 
                                num_heads, 
                                window_size,
                                mlp_ratio,
                                dropout,
                                attention_dropout,
                                stochastic_depth_prob,
                                norm_layer,
                                block,
                                use_checkpoint))
    def forward(self, x):
        for model in self.models:
            x = model(x)
        return x


if __name__ == "__main__":
    dim = 30
    window_size = 7
    shift_size = 2
    num_heads = 10
    model = ShiftedWindowAttention(dim, window_size, shift_size, num_heads)
    input_x = torch.zeros(4, 10, 30)
    y = model(input_x)
    print(y.shape)

    # SwinTransformer Block.
    dim = 96
    num_heads = 2
    input_x = torch.zeros(4, 250, 96)
    model = SwinTransformerBlock_1d(dim, num_heads, window_size, shift_size)
    y = model(input_x)
    print(f"The Swin block output shape is: {y.shape}")

    patch_size = 4
    input_dim = 30
    embed_dim = 96
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 7
    stochastic_depth_prob = 0.2
    model = Swin1dClass(
        patch_size,
        input_dim,
        embed_dim,
        depths,
        num_heads,
        window_size,
        stochastic_depth_prob=0.2,
        use_checkpoint=False,
    )
    input_x = torch.zeros(4, 1000, 30)
    y = model(input_x)
    print(f"swin1d_seq class output shape: {y.shape}")

    input_dim = 30
    embed_dim = 48
    output_dim = 128
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 8
    stochastic_depth_prob = 0.2
    model = Swin1dSeq(
        input_dim,
        embed_dim,
        output_dim,
        depths,
        num_heads,
        window_size,
        stochastic_depth_prob=0.2,
    )
    input_x = torch.zeros(1, int(1e5), 30)
    input_x = input_x.cuda()
    model.cuda()
    y = model(input_x)
    print(f"swin1d_seq output shape: {y.shape}")

    # multiple layer swin1d_seq
    num_layers = 2
    model = Swin1dSeqMultipleLayer(input_dim,
                                   embed_dim,
                                   output_dim,
                                   depths,
                                   num_heads,
                                   window_size,
                                   num_layers,
                                   stochastic_depth_prob=0.2)
    model.cuda()
    y = model(input_x)
    print(f"multiple layer implenmentation of swin1d_seq output shape: {y.shape}")