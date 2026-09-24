import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch.nn.functional as F


def weights_init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


from clip_cc.clip import clip


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model


class DepthwiseSeparableConv(nn.Module):
    """
    轻量局部对齐模块：depthwise(3×3 逐通道空间卷积) + pointwise(1×1 跨通道融合)。
    给 CLIP-ViT 的全局特征补上 CNN 的局部归纳偏置，强化空间结构、抑制衣物区域噪声。
    残差连接保留原始 ViT 全局语义，避免训练初期破坏预训练特征。
    """
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        # depthwise：groups=dim 使每个通道独立做空间卷积
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                            padding=padding, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        # pointwise：1×1 跨通道信息融合
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, D, h, w)
        identity = x
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw(x)
        return x + identity   # 残差连接


class TransReID(nn.Module):
    def __init__(self):
        super(TransReID, self).__init__()
        self.model_name = 'ViT-B-16'
        self.in_planes = 768
        self.in_planes_proj = 512

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        input_size_train = [256, 128]
        stride_size = [16, 16]
        self.h_resolution = int((input_size_train[0] - 16) // stride_size[0] + 1)
        self.w_resolution = int((input_size_train[1] - 16) // stride_size[1] + 1)
        self.vision_stride_size = stride_size[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        # Trick: freeze patch projection for improved stability
        # https://arxiv.org/pdf/2104.02057.pdf
        for _, v in self.image_encoder.conv1.named_parameters():
            v.requires_grad_(False)
        print('Freeze patch projection layer with shape {}'.format(self.image_encoder.conv1.weight.shape))
        # 768 3 16 16

        # ===== 新增：轻量局部对齐模块 =====
        self.local_conv = DepthwiseSeparableConv(self.in_planes, kernel_size=3)
        self.local_conv.apply(weights_init_kaiming)   # isinstance 判断后，仅对内部 Conv/BN 生效

        self.num_regions = 4

    def forward(self, x=None, cv_embed=None):
        cv_embed = None
        _, x12, xproj = self.image_encoder(x, cv_embed)

        # ===== 全局特征 (CLS token)，保持不变 =====
        image_features = x12[:, 0, :]
        image_features_proj = xproj[:, 0, :]

        feat_global = self.bottleneck(image_features)
        feat_proj = self.bottleneck_proj(image_features_proj)

        out_feat = torch.cat([feat_global, feat_proj], dim=1)
        out_feat = F.normalize(out_feat, dim=1)

        # ===== 局部特征提取（加入轻量卷积对齐）=====
        B, N, D = x12.shape
        h, w = self.h_resolution, self.w_resolution

        local_tokens = x12[:, 1:, :]               # 去掉 CLS token
        assert (h * w == N - 1), "Feature size mismatch."

        # 还原到 2D 空间，并转成卷积期望的 (B, D, h, w)
        local_features = local_tokens.reshape(B, h, w, D).permute(0, 3, 1, 2)  # (B, D, h, w)

        # 轻量卷积增强局部空间对齐
        local_features = self.local_conv(local_features)                       # (B, D, h, w)

        # 转回 (B, h, w, D) 以复用原有分块逻辑
        local_features = local_features.permute(0, 2, 3, 1)                    # (B, h, w, D)

        # 纵向分块 + 平均池化（暂保留固定分块；后续可替换为 TPM+PMG）
        local_regions = torch.chunk(local_features, self.num_regions, dim=1)
        local_feats = []
        for region in local_regions:
            region_feat = region.amax(dim=[1, 2])
            local_feats.append(region_feat)
        local_feats = torch.stack(local_feats, dim=1)                          # (B, num_regions, D)

        return out_feat, local_feats

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if not self.training and 'classifier' in i:
                continue  # ignore classifier weights in evaluation
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model():
    model = TransReID()
    return model