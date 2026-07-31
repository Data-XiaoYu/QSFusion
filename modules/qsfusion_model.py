import torch
import torch.nn as nn

from modules.quality_assessment import (
    ReferenceFreeQualityEstimator,
    reference_free_quality_target,
)
from modules.scale_interaction import MCIModule
from modules.semantic_encoder import SharedSemanticEncoder
from modules.semantic_guidance import ASGModule
from modules.transformer_backbone import (
    DualStreamCrossAttention,
    FeatureFusionProjection,
    GAEBlock,
    InfraredStreamEncoder,
    AWVAModule,
    UpsampleProjection,
    VisibleStreamEncoder,
)


class QSFusion(nn.Module):
    """Quality-aware and semantic-guided infrared-visible image fusion framework."""

    def __init__(
        self,
        in_visible_channels=3,
        in_infrared_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(2, 2, 2, 2),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2,
        bias=False,
        layer_norm_type="WithBias",
    ):
        super().__init__()

        self.asg = ASGModule(
            in_channels_ir=in_infrared_channels,
            in_channels_vi=in_visible_channels,
            mid_channels=32,
        )
        self.visible_stream_encoder = VisibleStreamEncoder(
            in_channels=in_visible_channels,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type,
        )
        self.infrared_stream_encoder = InfraredStreamEncoder(
            in_channels=in_infrared_channels,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type,
        )

        self.deep_cross_attention = DualStreamCrossAttention(dim * 2**3)
        self.deep_feature_fusion = FeatureFusionProjection(embed_dim=dim * 2**3)
        self.awva = AWVAModule(dim * 2**3)
        self.mci = MCIModule(channels=dim * 2**3, gamma=0.5)

        self.deep_decoder = nn.Sequential(
            *[
                GAEBlock(dim * 2**3, heads[3], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_blocks[3])
            ]
        )

        self.fusion_skip_l3 = FeatureFusionProjection(embed_dim=dim * 2**2)
        self.upsample_l4_to_l3 = UpsampleProjection(dim * 2**3)
        self.channel_projection_l3 = nn.Conv2d(dim * 2**3, dim * 2**2, 1, bias=bias)
        self.decoder_l3 = nn.Sequential(
            *[
                GAEBlock(dim * 2**2, heads[2], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_blocks[2])
            ]
        )

        self.fusion_skip_l2 = FeatureFusionProjection(embed_dim=dim * 2)
        self.upsample_l3_to_l2 = UpsampleProjection(dim * 2**2)
        self.channel_projection_l2 = nn.Conv2d(dim * 2**2, dim * 2, 1, bias=bias)
        self.decoder_l2 = nn.Sequential(
            *[
                GAEBlock(dim * 2, heads[1], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_blocks[1])
            ]
        )

        self.fusion_skip_l1 = FeatureFusionProjection(embed_dim=dim)
        self.upsample_l2_to_l1 = UpsampleProjection(dim * 2)
        self.decoder_l1 = nn.Sequential(
            *[
                GAEBlock(dim * 2, heads[0], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_blocks[0])
            ]
        )

        self.refinement_stage = nn.Sequential(
            *[
                GAEBlock(dim * 2, heads[0], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_refinement_blocks)
            ]
        )
        self.output_conv = nn.Conv2d(dim * 2, out_channels, 3, stride=1, padding=1, bias=bias)

        self.shared_semantic_encoder = SharedSemanticEncoder(
            in_channels=out_channels, base_channels=dim
        )
        self.reference_free_quality_estimator = ReferenceFreeQualityEstimator(
            in_channels=out_channels, base_channels=dim
        )

    def forward(self, infrared: torch.Tensor, visible: torch.Tensor):
        attention_mask = self.asg(infrared, visible)
        infrared_weight = attention_mask
        visible_weight = 1.0 - attention_mask

        residual_ratio = 0.05
        enhanced_infrared = infrared * infrared_weight + residual_ratio * infrared * (1.0 - infrared_weight)
        enhanced_visible = visible * visible_weight + residual_ratio * visible * (1.0 - visible_weight)

        vis_l4, vis_l3, vis_l2, vis_l1 = self.visible_stream_encoder(enhanced_visible)
        ir_l4, ir_l3, ir_l2, ir_l1 = self.infrared_stream_encoder(enhanced_infrared)

        vis_l4, ir_l4 = self.deep_cross_attention(vis_l4, ir_l4)
        deep_feature = self.deep_feature_fusion(vis_l4, ir_l4)
        deep_feature = self.awva(deep_feature)
        deep_feature = self.mci(deep_feature)

        decoded_l4 = self.deep_decoder(deep_feature)

        decoded_l3 = self.upsample_l4_to_l3(decoded_l4)
        skip_l3 = self.fusion_skip_l3(vis_l3, ir_l3)
        decoded_l3 = self.channel_projection_l3(torch.cat([decoded_l3, skip_l3], dim=1))
        decoded_l3 = self.decoder_l3(decoded_l3)

        decoded_l2 = self.upsample_l3_to_l2(decoded_l3)
        skip_l2 = self.fusion_skip_l2(vis_l2, ir_l2)
        decoded_l2 = self.channel_projection_l2(torch.cat([decoded_l2, skip_l2], dim=1))
        decoded_l2 = self.decoder_l2(decoded_l2)

        decoded_l1 = self.upsample_l2_to_l1(decoded_l2)
        skip_l1 = self.fusion_skip_l1(vis_l1, ir_l1)
        decoded_l1 = self.decoder_l1(torch.cat([decoded_l1, skip_l1], dim=1))
        refined_feature = self.refinement_stage(decoded_l1)

        fused = torch.sigmoid(self.output_conv(refined_feature))

        s_ir = self.shared_semantic_encoder(infrared)
        s_vi = self.shared_semantic_encoder(visible)
        s_fused = self.shared_semantic_encoder(fused)
        q_pred = self.reference_free_quality_estimator(fused)
        q_ir = reference_free_quality_target(infrared)
        q_vi = reference_free_quality_target(visible)
        q_star = torch.maximum(q_ir, q_vi).detach()

        return fused, s_ir, s_vi, s_fused, q_pred, q_star


def build_qsfusion(
    dim=48,
    num_blocks=(2, 2, 2, 2),
    num_refinement_blocks=4,
    heads=(1, 2, 4, 8),
) -> QSFusion:
    return QSFusion(
        dim=dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refinement_blocks,
        heads=heads,
    )
