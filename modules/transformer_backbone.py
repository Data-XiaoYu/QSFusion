import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _to_sequence(x):
    return rearrange(x, "b c h w -> b (h w) c")


def _to_feature_map(x, height, width):
    return rearrange(x, "b (h w) c -> b c h w", h=height, w=width)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))

    def forward(self, x):
        variance = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(variance + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class FeatureMapLayerNorm(nn.Module):
    def __init__(self, dim, layer_norm_type):
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x):
        height, width = x.shape[-2:]
        return _to_feature_map(self.body(_to_sequence(x)), height, width)


class CFFN(nn.Module):
    """Convolutional feed-forward network used in the GAE block."""

    def __init__(self, dim, expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * expansion_factor)
        self.input_projection = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.depthwise_conv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.output_projection = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.depthwise_conv(x)
        x = F.gelu(x)
        return self.output_projection(x)


class MDTA(nn.Module):
    """Multi-DConv head transposed self-attention in the GAE block."""

    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv_projection = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_depthwise_conv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.output_projection = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, height, width = x.shape
        qkv = self.qkv_depthwise_conv(self.qkv_projection(x))
        query, key, value = qkv.chunk(3, dim=1)

        query = rearrange(query, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        key = rearrange(key, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        value = rearrange(value, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        attention = (query @ key.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)

        out = attention @ value
        out = rearrange(
            out,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=height,
            w=width,
        )
        return self.output_projection(out)


class GAEBlock(nn.Module):
    """Global-aware enhancement block composed of MDTA and CFFN."""

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, layer_norm_type):
        super().__init__()
        self.mdta_norm = FeatureMapLayerNorm(dim, layer_norm_type)
        self.mdta = MDTA(dim, num_heads, bias)
        self.cffn_norm = FeatureMapLayerNorm(dim, layer_norm_type)
        self.cffn = CFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.mdta(self.mdta_norm(x))
        x = x + self.cffn(self.cffn_norm(x))
        return x


class OverlappedPatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=48, bias=False):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, x):
        return self.projection(x)


class DownsampleProjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.projection(x)


class UpsampleProjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.projection(x)


class VisibleStreamEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        dim=32,
        num_blocks=(2, 3, 3, 4),
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
        layer_norm_type="WithBias",
    ):
        super().__init__()
        self.patch_projection = OverlappedPatchEmbedding(in_channels, dim, bias)
        self.level1_gae = nn.Sequential(
            *[
                GAEBlock(dim, heads[0], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_blocks[0])
            ]
        )
        self.downsample_1_to_2 = DownsampleProjection(dim)
        self.level2_gae = nn.Sequential(
            *[
                GAEBlock(dim * 2, heads[1], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_blocks[1])
            ]
        )
        self.downsample_2_to_3 = DownsampleProjection(dim * 2)
        self.level3_gae = nn.Sequential(
            *[
                GAEBlock(dim * 4, heads[2], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_blocks[2])
            ]
        )
        self.downsample_3_to_4 = DownsampleProjection(dim * 4)
        self.level4_gae = nn.Sequential(
            *[
                GAEBlock(dim * 8, heads[3], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_blocks[3])
            ]
        )

    def forward(self, visible):
        level1 = self.level1_gae(self.patch_projection(visible))
        level2 = self.level2_gae(self.downsample_1_to_2(level1))
        level3 = self.level3_gae(self.downsample_2_to_3(level2))
        level4 = self.level4_gae(self.downsample_3_to_4(level3))
        return level4, level3, level2, level1


class InfraredStreamEncoder(VisibleStreamEncoder):
    pass


class DualStreamCrossAttention(nn.Module):
    def __init__(self, channels, num_heads=1, norm_groups=16):
        super().__init__()
        self.num_heads = num_heads
        self.visible_norm = nn.GroupNorm(norm_groups, channels)
        self.infrared_norm = nn.GroupNorm(norm_groups, channels)
        self.visible_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.visible_output = nn.Conv2d(channels, channels, 1)
        self.infrared_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.infrared_output = nn.Conv2d(channels, channels, 1)

    def forward(self, visible_feature, infrared_feature):
        batch, channels, height, width = visible_feature.shape
        head_dim = channels // self.num_heads

        visible_norm = self.visible_norm(visible_feature)
        visible_qkv = self.visible_qkv(visible_norm).view(
            batch, self.num_heads, head_dim * 3, height, width
        )
        visible_query, visible_key, visible_value = visible_qkv.chunk(3, dim=2)

        infrared_norm = self.infrared_norm(infrared_feature)
        infrared_qkv = self.infrared_qkv(infrared_norm).view(
            batch, self.num_heads, head_dim * 3, height, width
        )
        infrared_query, infrared_key, infrared_value = infrared_qkv.chunk(3, dim=2)

        visible_attention = torch.einsum(
            "bnchw, bncyx -> bnhwyx", infrared_query, visible_key
        ).contiguous() / math.sqrt(channels)
        visible_attention = visible_attention.view(batch, self.num_heads, height, width, -1)
        visible_attention = torch.softmax(visible_attention, -1)
        visible_attention = visible_attention.view(
            batch, self.num_heads, height, width, height, width
        )
        visible_out = torch.einsum(
            "bnhwyx, bncyx -> bnchw", visible_attention, visible_value
        ).contiguous()
        visible_out = self.visible_output(visible_out.view(batch, channels, height, width))

        infrared_attention = torch.einsum(
            "bnchw, bncyx -> bnhwyx", visible_query, infrared_key
        ).contiguous() / math.sqrt(channels)
        infrared_attention = infrared_attention.view(batch, self.num_heads, height, width, -1)
        infrared_attention = torch.softmax(infrared_attention, -1)
        infrared_attention = infrared_attention.view(
            batch, self.num_heads, height, width, height, width
        )
        infrared_out = torch.einsum(
            "bnhwyx, bncyx -> bnchw", infrared_attention, infrared_value
        ).contiguous()
        infrared_out = self.infrared_output(infrared_out.view(batch, channels, height, width))

        return visible_out + visible_norm, infrared_out + infrared_norm


class AWVAModule(nn.Module):
    """Attention weighted value aggregation used after deep feature fusion."""

    def __init__(self, channels, num_heads=1, norm_groups=16):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(norm_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, feature):
        batch, channels, height, width = feature.shape
        head_dim = channels // self.num_heads
        normed = self.norm(feature)
        qkv = self.qkv(normed).view(batch, self.num_heads, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)
        attention = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channels)
        attention = attention.view(batch, self.num_heads, height, width, -1)
        attention = torch.softmax(attention, -1)
        attention = attention.view(batch, self.num_heads, height, width, height, width)
        out = torch.einsum("bnhwyx, bncyx -> bnchw", attention, value).contiguous()
        out = self.output(out.view(batch, channels, height, width))
        return out + feature


class FeatureFusionProjection(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super().__init__()
        self.projection = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, visible_feature, infrared_feature):
        return self.projection(torch.cat([visible_feature, infrared_feature], dim=1))
