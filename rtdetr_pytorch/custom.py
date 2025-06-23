import os
import sys

import torch
import torch.nn as nn
import onnx
import onnxruntime as rt

from src.nn.backbone import PResNet
from src.zoo.rtdetr import HybridEncoder
from src.zoo.rtdetr import RTDETRTransformer

weights_path = '/home/kira9k/Загрузки/ResNet18_vd_pretrained_from_paddle.pth'


def generate_anchors(spatial_shapes=None,
                     grid_size=0.05,
                     dtype=torch.float32,
                     device='cpu'):
    spatial_shapes = [[80, 80], [40, 40], [20, 20]]
    anchors = []
    for lvl, (h, w) in enumerate(spatial_shapes):
        grid_y, grid_x = torch.meshgrid(\
            torch.arange(end=h, dtype=dtype), \
            torch.arange(end=w, dtype=dtype), indexing='ij')
        grid_xy = torch.stack([grid_x, grid_y], -1)
        valid_WH = torch.tensor([w, h]).to(dtype)
        grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
        wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
        anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))
    anchors = torch.concat(anchors, 1).to(device)
    valid_mask = ((anchors > 0.01) * (anchors < 1 - 0.01)).all(-1,
                                                               keepdim=True)
    anchors = torch.log(anchors / (1 - anchors))
    # anchors = torch.where(valid_mask, anchors, float('inf'))
    # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
    anchors = torch.where(valid_mask, anchors, torch.inf)
    valid_mask = valid_mask.to(torch.float32)
    valid_mask = torch.cat([valid_mask] * 256, dim=2)
    return anchors, valid_mask


anchors, valid_mask = generate_anchors()
print(anchors.shape, valid_mask.shape)
model_fpn = PResNet(depth=18,
                    variant='d',
                    num_stages=4,
                    return_idx=[1, 2, 3],
                    act='relu',
                    freeze_at=-1,
                    freeze_norm=False,
                    pretrained=True)

model_fpn.load_state_dict(torch.load(weights_path, weights_only=True))

model_enc = HybridEncoder(in_channels=[128, 256, 512],
                          feat_strides=[8, 16, 32],
                          hidden_dim=256,
                          nhead=8,
                          dim_feedforward=1024,
                          dropout=0.0,
                          enc_act='gelu',
                          use_encoder_idx=[2],
                          num_encoder_layers=1,
                          pe_temperature=10000,
                          expansion=0.5,
                          depth_mult=1,
                          act='silu',
                          eval_spatial_size=[640, 640])

model_transformer_decoder = RTDETRTransformer(num_classes=1,
                                              hidden_dim=256,
                                              num_queries=300,
                                              position_embed_type='sine',
                                              feat_channels=[256, 256, 256],
                                              feat_strides=[8, 16, 32],
                                              num_levels=3,
                                              num_decoder_points=3,
                                              nhead=8,
                                              num_decoder_layers=3,
                                              dim_feedforward=1024,
                                              dropout=0.,
                                              activation="relu",
                                              num_denoising=100,
                                              label_noise_ratio=0.5,
                                              box_noise_scale=1.0,
                                              learnt_init_query=False,
                                              eval_spatial_size=[640, 640],
                                              eval_idx=-1,
                                              eps=1e-2,
                                              aux_loss=True)

data = torch.rand(1, 3, 640, 640, dtype=torch.float32)
#size = torch.tensor([[640., 640.]])


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.backbone = model_fpn
        self.encoder = model_enc
        self.decoder = model_transformer_decoder

    def forward(self, x, anchors, valid_mask):
        feats = self.backbone(x)
        print(f"Backbone output: {[f.shape for f in feats]}")
        enc = self.encoder(feats)
        print(f"Enc output: {[f.shape for f in enc]}")
        x = self.decoder(enc, anchors=anchors, valid_mask=valid_mask)
        return x


model = Model()
#checkpoint = torch.load('output/rtdetr_r18vd_6x_coco/checkpoint.pth')
#state = checkpoint['ema']['module']
#model.load_state_dict(state)
model.eval()

data = torch.rand(1, 3, 640, 640)
with torch.no_grad():
    torch.onnx.export(
        model,
        (data, anchors, valid_mask),
        'model.onnx',
        input_names=['images', 'anchors',
                     'valid_mask'],  # Имена для всех входов
        output_names=['labels', 'boxes'],  # Имена выходов
        opset_version=16,  # Для широкой совместимости
        do_constant_folding=True,
        verbose=False)

model_onnx = onnx.load("model.onnx")
onnx.checker.check_model(model_onnx)
print("Inputs:", [input.name for input in model_onnx.graph.input])
print("Outputs:", [output.name for output in model_onnx.graph.output])

import onnxsim

model_simp, check = onnxsim.simplify("model.onnx")
if check:
    onnx.save(model_simp, "model.onnx")
    print("Simplified ONNX model saved!")
else:
    print("Simplification failed!")
