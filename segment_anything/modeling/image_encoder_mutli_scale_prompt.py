# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from itertools import repeat
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from segment_anything.modeling.common import LayerNorm2d, MLPBlock
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        peft=None
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                num=i,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                peft=peft if (i % 1==0) else None,
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        # Construct multi-scale inputs and fuse positions`
        self.peft_index = [1,3,5,7,9,11]
        fusionCounts = len(self.peft_index)
        groupConCounts = 12
        group_num = [4, 8, 16]
        # multi_adapter
        self.specific_VitCnnFusions = nn.ModuleList()
        # Multi-scale fusion module
        self.specific_fpnFusion = FPNMultiScaleFusionModule()
        # Multi-scale generation module
        self.specific_fpnGenerate = FPNMultiScaleGenerateModule()
        # Fusion count`
        for i in range(fusionCounts):
            self.specific_VitCnnFusions.append(MutliAdapter())
        self.specific_groupConvs_2 = nn.ModuleList()
        self.specific_groupConvs_4 = nn.ModuleList()
        self.specific_groupConvs_8 = nn.ModuleList()
        for i in range(groupConCounts):
            self.specific_groupConvs_2.append(GroupConv(12,12,group_num[0]))
            self.specific_groupConvs_4.append(GroupConv(48,48,group_num[1]))
            self.specific_groupConvs_8.append(GroupConv(192,192,group_num[2]))

        self.apply(self._init_weights)

        self.specific_spgen = SPGen()

        self.up1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, origin, apply_drop=False):

        x_2, x_4, x_8 = self.specific_fpnGenerate(origin)
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        each_depth = []
        fusion_index = 0
        for i, (blk,gc2,gc4,gc8) in enumerate(zip(self.blocks, self.specific_groupConvs_2,self.specific_groupConvs_4,self.specific_groupConvs_8)):
            if (i in self.peft_index):
                x_con_2 = gc2(x_2)
                x_con_4 = gc4(x_4)
                x_con_8 = gc8(x_8)
                x_vit = blk(x).permute(0, 3, 1, 2)
                x_vit_out, x_cnn_2_out, x_cnn_4_out, x_cnn_8_out = self.specific_VitCnnFusions[fusion_index](x_vit,x_con_2,x_con_4,x_con_8)
                fusion_index += 1
                x = (x_vit + x_vit_out).permute(0, 2, 3, 1)
                x_2 = x_con_2 + x_cnn_2_out
                x_4 = x_con_4 + x_cnn_4_out
                x_8 = x_con_8 + x_cnn_8_out
            else:
                x_2 = gc2(x_2)
                x_4 = gc4(x_4)
                x_8 = gc8(x_8)
                x = blk(x)
            x_ = self.neck(x.permute(0, 3, 1, 2))
            each_depth.append(x_)


        stacked_embedding = torch.stack(each_depth, dim=0)


        cnn_out = self.specific_fpnFusion(x_8, x_4, x_2)
        if apply_drop:
            spgen1, spgen2, spgen3, spgen4 = self.specific_spgen(stacked_embedding, apply_drop=False)
            n_samples = 10
            predictions = []
            with torch.no_grad():
                for i in range(n_samples):
                    unc_spgen1, unc_spgen2, unc_spgen3, unc_spgen4 = self.specific_spgen(stacked_embedding, apply_drop=True)
                    unc_spgen1 = self.up1(unc_spgen1)
                    unc_spgen2 = self.up2(unc_spgen2)
                    unc_spgen3 = self.up3(unc_spgen3)
                    output = postprocess_masks(unc_spgen1 + unc_spgen2 + unc_spgen3 + unc_spgen4)
                    output = torch.sigmoid(output).detach()
                    predictions.append(output)
            predictions = torch.stack(predictions)
            uncertainty = torch.var(predictions, dim=0)
            main_uncertainty = normalize_uncertainty(uncertainty, method=' ').to('cuda')
            return each_depth[-1], spgen1, spgen2, spgen3, spgen4, cnn_out, main_uncertainty

        else:
            spgen1, spgen2, spgen3, spgen4 = self.specific_spgen(stacked_embedding, apply_drop)
            return each_depth[-1], spgen1, spgen2, spgen3, spgen4, cnn_out

def normalize_uncertainty(uncertainty, method='log'):

    if method == 'log':
        uncertainty = torch.log(uncertainty + 1e-10)
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
    elif method == 'zscore':
        mean = uncertainty.mean()
        std = uncertainty.std()
        uncertainty = (uncertainty - mean) / std
    else:
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
    return uncertainty

def postprocess_masks(
        masks,
        input_size=(256, 256),
        original_size=(512, 512),
):
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks

class FPNMultiScaleGenerateModule(nn.Module):
    def __init__(self):
        super(FPNMultiScaleGenerateModule, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1),
            LayerNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(12, 48, kernel_size=3, padding=1),
            LayerNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(48, 192, kernel_size=3, padding=1),
            LayerNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.lateral1 = ASPP(12,12)
        self.lateral2 = ASPP(48,48)
        self.lateral3 = ASPP(192,192)

        self.up1 = nn.Sequential(nn.ConvTranspose2d(192, 48, kernel_size=2, stride=2),
                                 LayerNorm2d(48),
                                 nn.GELU())

        self.up2 = nn.Sequential(nn.ConvTranspose2d(48, 12, kernel_size=2, stride=2),
                                 LayerNorm2d(12),
                                 nn.GELU())


    def forward(self, x):
        x1 = self.block1(x)  # (1, 3, 512, 512) -> (1, 12, 256, 256)
        x2 = self.block2(x1)  # (1, 12, 256, 256) -> (1, 48, 128, 128)
        x3 = self.block3(x2)  # (1, 48, 128, 128) -> (1, 192, 64, 64)

        x8_out = self.lateral3(x3)
        x4_out = self.up1(x8_out) + self.lateral2(x2)
        x2_out = self.up2(x4_out) + self.lateral1(x1)

        return x2_out, x4_out, x8_out



class FPNMultiScaleFusionModule(nn.Module):
    def __init__(self):
        super(FPNMultiScaleFusionModule, self).__init__()

        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.lateral1 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=1),
                                      LayerNorm2d(64),
                                      nn.GELU()
                                      )
        self.lateral2 = nn.Sequential(nn.Conv2d(48, 64, kernel_size=1),
                                      LayerNorm2d(64),
                                      nn.GELU()
                                      )
        self.lateral3 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1),
                                      LayerNorm2d(64),
                                      nn.GELU()
                                      )

        self.conv_final = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 输出为单通道
                                        LayerNorm2d(32),
                                        nn.GELU(),
                                        nn.Conv2d(32, 1, kernel_size=3, padding=1),

                                        )

    def forward(self, x1, x2, x3):

        x1 = self.lateral1(x1)
        x1_up = self.upsample2x(x1)

        x2 = self.lateral2(x2)
        x2_up = x2 + x1_up

        x3 = self.lateral3(x3)
        x3_up = x3 + self.upsample2x(x2_up)
        x3_up = self.upsample2x(x3_up)

        fused = x3_up
        output = self.conv_final(fused)

        return output

class DownsampleModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(DownsampleModule, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.ln = LayerNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.ln(x)
        x = self.relu(x)

        return x
class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding)
        self.ln1 = LayerNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding)
        self.ln2 = LayerNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding)
        self.ln3 = LayerNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.ln2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.ln3(x)
        x = self.relu3(x)

        return x

class MutliAdapter(nn.Module):
    def __init__(self):
        super(MutliAdapter, self).__init__()

        self.relu = nn.ReLU()

        self.vit_conv = nn.Sequential(nn.Conv2d(768, 192, kernel_size=1, stride=1),
                                      LayerNorm2d(192),
                                      nn.GELU())

        self.up1 = nn.Sequential(nn.ConvTranspose2d(192, 48, kernel_size=2, stride=2),
                                 LayerNorm2d(48),
                                 nn.GELU())

        self.up2 = nn.Sequential(nn.ConvTranspose2d(48, 12, kernel_size=2, stride=2),
                                 LayerNorm2d(12),
                                 nn.GELU())

        self.conv_2 = nn.Sequential(nn.Conv2d(24, 12, kernel_size=1, stride=1),
                                      LayerNorm2d(12),
                                      nn.GELU())
        self.conv_4 = nn.Sequential(nn.Conv2d(96, 48, kernel_size=1, stride=1),
                                    LayerNorm2d(48),
                                    nn.GELU())
        self.conv_8 = nn.Sequential(nn.Conv2d(384, 192, kernel_size=1, stride=1),
                                    LayerNorm2d(192),
                                    nn.GELU())
        self.cbam_vit = CBAM(768)

        self.down1 = nn.Sequential(nn.Conv2d(24, 96, kernel_size=2, stride=2),
                                   LayerNorm2d(96),
                                   nn.GELU())
        self.down2 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=2, stride=2),
                                   LayerNorm2d(384),
                                   nn.GELU())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, vit_8, cnn_2, cnn_4,cnn_8):
        vit_8 = self.vit_conv(vit_8)

        vit_4 = self.up1(vit_8)
        vit_2 = self.up2(vit_4)

        x_fusion_2 = torch.cat((vit_2, cnn_2), dim=1)
        x_fusion_4 = torch.cat((vit_4, cnn_4), dim=1)
        x_fusion_8 = torch.cat((vit_8, cnn_8), dim=1)

        x_fusion_cnn_2 = self.conv_2(x_fusion_2)
        x_fusion_cnn_4 = self.conv_4(x_fusion_4)
        x_fusion_cnn_8 = self.conv_8(x_fusion_8)

        x_down_4 = self.down1(x_fusion_2)
        x_cat_4 = torch.cat((x_fusion_4, x_down_4), dim=1)
        x_down_8 = self.down2(x_cat_4)
        x_cat_8 = torch.cat((x_fusion_8, x_down_8), dim=1)
        x_fusion_vit = self.cbam_vit(x_cat_8)
        return x_fusion_vit, x_fusion_cnn_2, x_fusion_cnn_4, x_fusion_cnn_8

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=12):
        super(SEBlock, self).__init__()
        assert channels % reduction == 0, "channels must be divisible by reduction"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc2(F.relu(self.fc1(y)))
        return x * self.sigmoid(y).view(b, c, 1, 1)

class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.channel_attention = SEBlock(channels)
        self.spatial_attention = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.channel_attention(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        x_out = self.spatial_attention(x_out)
        return x * torch.sigmoid(x_out)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.dilationConv_0 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1),stride=(1,1))
        self.dilationConv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3),stride=(1,1),padding=3, dilation=(3,3))
        self.dilationConv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3),stride=(1,1),padding=5, dilation=(5,5))
        self.dilationConv_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3),stride=(1,1),padding=7, dilation=(7,7))
        self.dilationConv_fusion = nn.Conv2d(4*out_channels, out_channels, kernel_size=(1,1),stride=(1,1),padding=0, dilation=(1,1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x0 = F.relu(self.dilationConv_0(x))
        x1 = F.relu(self.dilationConv_1(x))
        x2 = F.relu(self.dilationConv_2(x))
        x3 = F.relu(self.dilationConv_3(x))
        x_cat = torch.cat([x0,x1,x2,x3], dim=1)
        x = F.relu(self.dilationConv_fusion(x_cat))
        return x

class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        peft: str = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num = num
        self.interval = 2
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            window_size=window_size,
            peft=peft
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size
        self.peft = peft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        window_size : int = 0,
        peft: str= None
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
        self.peft = peft
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)

        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        # if self.use_rel_pos and self.peft!="Prefix" and self.peft!="Uni—peft":
        #     attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        if self.use_rel_pos and self.peft!="Prefix" and self.peft!="Uni—peft":
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_origin = x
        x = self.proj(x)
        # x_fft = self.proj(x_fft)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        # x_fft = x_fft.permute(0, 2, 3, 1)
        return x


class GroupConv(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups):
        super(GroupConv, self).__init__()
        assert in_ch % num_groups == 0, "in_ch must be divisible by num_groups"
        assert out_ch % num_groups == 0, "out_ch must be divisible by num_groups"

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=num_groups, bias=False),
            LayerNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)



class PatchEmbed2(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # x = F.interpolate(x, size=2*x.shape[-1], mode='bilinear', align_corners=True)
        x = self.proj(x)
        return x



def to_2tuple(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 2))

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor




class SPGen(nn.Module):
    def __init__(self):
        super(SPGen, self).__init__()

        self.up1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=2, stride=2),
                                 LayerNorm2d(64),
                                 nn.GELU(),
                                 nn.Conv2d(64, 16, kernel_size=1, stride=1),
                                 LayerNorm2d(16),
                                 nn.GELU(),
                                 nn.Conv2d(16, 1, kernel_size=1))

        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=1, stride=1),
                                 LayerNorm2d(64),
                                 nn.GELU(),
                                 nn.ConvTranspose2d(64, 16, kernel_size=1, stride=1),
                                 LayerNorm2d(16),
                                 nn.GELU(),
                                 nn.Conv2d(16, 1, kernel_size=1))

        self.up3 = nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
                                 LayerNorm2d(64),
                                 nn.GELU(),
                                 nn.ConvTranspose2d(64, 16, kernel_size=1, stride=1),
                                 LayerNorm2d(16),
                                 nn.GELU(),
                                 nn.Conv2d(16, 1, kernel_size=1))

        self.up4 = nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
                                 LayerNorm2d(64),
                                 nn.GELU(),
                                 nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2),
                                 LayerNorm2d(16),
                                 nn.GELU(),
                                 nn.Conv2d(16, 1, kernel_size=1))


    def forward(self, x, apply_drop=False):
        if apply_drop:

            x1 = x[2]
            x2 = x[5]
            x3 = x[8]
            x4 = x[11]
            x1 = self.up1[:3](x1)
            x1 = F.dropout(x1, p=0.3, training=True)
            x1 = self.up1[3:6](x1)
            x1 = F.dropout(x1, p=0.3, training=True)
            x1 = self.up1[6:](x1)

            x2 = self.up2[:3](x2)
            x2 = F.dropout(x2, p=0.3, training=True)
            x2 = self.up2[3:6](x2)
            x2 = F.dropout(x2, p=0.3, training=True)
            x2 = self.up2[6:](x2)

            x3 = self.up3[:3](x3)
            x3 = F.dropout(x3, p=0.3, training=True)
            x3 = self.up3[3:6](x3)
            x3 = F.dropout(x3, p=0.3, training=True)
            x3 = self.up3[6:](x3)


            x4 = self.up4[:3](x4)
            x4 = F.dropout(x4, p=0.3, training=True)
            x4 = self.up4[3:6](x4)
            x4 = F.dropout(x4, p=0.3, training=True)
            x4 = self.up4[6:](x4)

            return x1, x2, x3, x4
        else:
            x1 = x[2]
            x2 = x[5]
            x3 = x[8]
            x4 = x[11]

            x1 = self.up1(x1)
            x2 = self.up2(x2)
            x3 = self.up3(x3)
            x4 = self.up4(x4)

            return x1, x2, x3, x4
