# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.modeling.backbone.fpn import FPN, LastLevelP6P7, LastLevelMaxPool
from timm.models import create_model


class RegionViTBackbone(Backbone):
    def __init__(self, model_name, freeze_at, drop_path_rate, det_norm):
        super().__init__()
        # create your own backbone, do I need to remove fc?
        self.model = create_model(model_name, pretrained=True, drop_path_rate=drop_path_rate, det_norm=det_norm)

        self._output_shape = [(self.model.embed_dim[1], 4), (self.model.embed_dim[2], 8),
                              (self.model.embed_dim[3], 16), (self.model.embed_dim[4], 32)]

        # freeze
        frozen_layers = []
        if freeze_at > 1:
            for name, parameter in self.model.named_parameters():
                if name.startswith('patch_embed') or name.startswith('cls_token'):
                    parameter.requires_grad_(False)
                    frozen_layers.append(name)
                if name.startswith('layers.0'):
                    parameter.requires_grad_(False)
                    frozen_layers.append(name)
        elif freeze_at > 0:
            for name, parameter in self.model.named_parameters():
                if name.startswith('patch_embed') or name.startswith('cls_token'):
                    parameter.requires_grad_(False)
                    frozen_layers.append(name)

        if len(frozen_layers) > 0:
            print("Following paramteres are frozen: ")
            for name in frozen_layers:
                print(name)
            print("==================================")

        del self.model.norm
        del self.model.head

    def forward(self, image):

        stage_outs = self.model.forward_features(image, detection=True)
        out = {}
        #print("i", image.shape)
        for i, x in enumerate(stage_outs, start=2):
            """
            if x.shape[-1] % 2 != 0 and x.shape[-2] % 2 != 0:
                x = nn.functional.pad(x, (0, 1, 0, 1))
            elif x.shape[-1] % 2 != 0:
                x = nn.functional.pad(x, (0, 0, 0, 1))
            elif x.shape[-2] % 2 != 0:
                x = nn.functional.pad(x, (0, 1, 0, 0))
            """
            out[f'res{i}'] = x
            #print(f'res{i}: {x.shape}')
        
        return out

    def output_shape(self):
        out = {}
        for i, x in enumerate(self._output_shape, start=2):
            out[f'res{i}'] = ShapeSpec(channels=x[0], stride=x[1])
        return out


@BACKBONE_REGISTRY.register()
def build_regionvit_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        LaViTBackbone: a :class:`ResNet` instance.
    """
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    model_name = cfg.MODEL.BACKBONE.REGIONVIT
    drop_path_rate = cfg.MODEL.BACKBONE.DP_RATE
    det_norm = cfg.MODEL.BACKBONE.DET_NORM
    model = RegionViTBackbone(model_name, freeze_at, drop_path_rate, det_norm)
    return model


@BACKBONE_REGISTRY.register()
def build_regionvit_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_regionvit_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_retinanet_regionvit_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_regionvit_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
