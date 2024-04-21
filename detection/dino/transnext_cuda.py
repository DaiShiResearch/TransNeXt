'''
TransNeXt: Robust Foveal Visual Perception for Vision Transformers
Paper: https://arxiv.org/abs/2311.17132
Code: https://github.com/DaiShiResearch/TransNeXt

Author: Dai Shi
Github: https://github.com/DaiShiResearch
Email: daishiresearch@gmail.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from mmdet.registry import MODELS
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import load_checkpoint
import swattention

CUDA_NUM_THREADS = 128


class sw_qkrpb_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, rpb, height, width, kernel_size):
        attn_weight = swattention.qk_rpb_forward(query, key, rpb, height, width, kernel_size, CUDA_NUM_THREADS)

        ctx.save_for_backward(query, key)
        ctx.height, ctx.width, ctx.kernel_size = height, width, kernel_size

        return attn_weight

    @staticmethod
    def backward(ctx, d_attn_weight):
        query, key = ctx.saved_tensors
        height, width, kernel_size = ctx.height, ctx.width, ctx.kernel_size

        d_query, d_key, d_rpb = swattention.qk_rpb_backward(d_attn_weight.contiguous(), query, key, height, width,
                                                            kernel_size, CUDA_NUM_THREADS)

        return d_query, d_key, d_rpb, None, None, None


class sw_av_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_weight, value, height, width, kernel_size):
        output = swattention.av_forward(attn_weight, value, height, width, kernel_size, CUDA_NUM_THREADS)

        ctx.save_for_backward(attn_weight, value)
        ctx.height, ctx.width, ctx.kernel_size = height, width, kernel_size

        return output

    @staticmethod
    def backward(ctx, d_output):
        attn_weight, value = ctx.saved_tensors
        height, width, kernel_size = ctx.height, ctx.width, ctx.kernel_size

        d_attn_weight, d_value = swattention.av_backward(d_output.contiguous(), attn_weight, value, height, width,
                                                         kernel_size, CUDA_NUM_THREADS)

        return d_attn_weight, d_value, None, None, None


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@torch.no_grad()
def get_relative_position_cpb(query_size, key_size, pretrain_size=None,
                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)
    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw)
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw)

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)

    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))

    return idx_map, relative_coords_table


@torch.no_grad()
def get_seqlen_scale(input_resolution, window_size, device):
    return torch.nn.functional.avg_pool2d(
        torch.ones(1, input_resolution[0], input_resolution[1], device=device) * (window_size ** 2), window_size,
        stride=1, padding=window_size // 2, ).reshape(-1, 1)


class AggregatedAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1, is_extrapolation=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.sr_ratio = sr_ratio

        self.is_extrapolation = is_extrapolation

        if not is_extrapolation:
            # The estimated training resolution is used for bilinear interpolation of the generated relative position bias.
            self.trained_H, self.trained_W = input_resolution
            self.trained_len = self.trained_H * self.trained_W
            self.trained_pool_H, self.trained_pool_W = input_resolution[0] // self.sr_ratio, input_resolution[
                1] // self.sr_ratio
            self.trained_pool_len = self.trained_pool_H * self.trained_pool_W

        assert window_size % 2 == 1, "window size must be odd"
        self.window_size = window_size
        self.local_len = window_size ** 2

        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        # relative_bias_local:
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0,
                                  std=0.0004))

        # dynamic_local_bias:
        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale):
        B, N, C = x.shape
        pool_H, pool_W = H // self.sr_ratio, W // self.sr_ratio
        pool_len = pool_H * pool_W

        # Generate queries, normalize them with L2, add query embedding, and then magnify with sequence length scale and temperature.
        # Use softplus function ensuring that the temperature is not lower than 0.
        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * seq_length_scale

        # Generate unfolded keys and values and l2-normalize them
        k_local, v_local = self.kv(x).reshape(B, N, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3).chunk(2,
                                                                                                                 dim=1)

        # Compute local similarity
        attn_local = sw_qkrpb_cuda.apply(q_norm_scaled.contiguous(), F.normalize(k_local, dim=-1).contiguous(),
                                         self.relative_pos_bias_local,
                                         H, W, self.window_size)

        # Generate pooled features
        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = F.adaptive_avg_pool2d(self.act(self.sr(x_)), (pool_H, pool_W)).reshape(B, -1, pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        # Generate pooled keys and values
        kv_pool = self.kv(x_).reshape(B, pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        if self.is_extrapolation:
            ##Use MLP to generate continuous relative positional bias for pooled features.
            pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                        relative_pos_index.view(-1)].view(-1, N, pool_len)
        else:
            ##Use MLP to generate continuous relative positional bias for pooled features.
            pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                        relative_pos_index.view(-1)].view(-1, self.trained_len, self.trained_pool_len)

            # bilinear interpolation:
            pool_bias = pool_bias.reshape(-1, self.trained_len, self.trained_pool_H, self.trained_pool_W)
            pool_bias = F.interpolate(pool_bias, (pool_H, pool_W), mode='bilinear')
            pool_bias = pool_bias.reshape(-1, self.trained_len, pool_len).transpose(-1, -2).reshape(-1, pool_len,
                                                                                                    self.trained_H,
                                                                                                    self.trained_W)
            pool_bias = F.interpolate(pool_bias, (H, W), mode='bilinear').reshape(-1, pool_len, N).transpose(-1, -2)

        # Compute pooled similarity
        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias

        # Concatenate local & pooled similarity matrices and calculate attention weights through the same Softmax
        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Split the attention weights and separately aggregate the values of local & pooled features
        attn_local, attn_pool = torch.split(attn, [self.local_len, pool_len], dim=-1)
        attn_local = (q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local
        x_local = sw_av_cuda.apply(attn_local.type_as(v_local), v_local.contiguous(), H, W, self.window_size)
        x_pool = attn_pool @ v_pool
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=True, attn_drop=0.,
                 proj_drop=0., is_extrapolation=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.is_extrapolation = is_extrapolation

        if not is_extrapolation:
            # The estimated training resolution is used for bilinear interpolation of the generated relative position bias.
            self.trained_H, self.trained_W = input_resolution
            self.trained_len = self.trained_H * self.trained_W

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=1)

        if self.is_extrapolation:
            # Use MLP to generate continuous relative positional bias
            rel_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                       relative_pos_index.view(-1)].view(-1, N, N)
        else:
            # Use MLP to generate continuous relative positional bias
            rel_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                       relative_pos_index.view(-1)].view(-1, self.trained_len, self.trained_len)
            # bilinear interpolation:
            rel_bias = rel_bias.reshape(-1, self.trained_len, self.trained_H, self.trained_W)
            rel_bias = F.interpolate(rel_bias, (H, W), mode='bilinear')
            rel_bias = rel_bias.reshape(-1, self.trained_len, N).transpose(-1, -2).reshape(-1, N, self.trained_H,
                                                                                           self.trained_W)
            rel_bias = F.interpolate(rel_bias, (H, W), mode='bilinear').reshape(-1, N, N).transpose(-1, -2)

        attn = ((F.normalize(q, dim=-1) + self.query_embedding) * F.softplus(
            self.temperature) * seq_length_scale) @ F.normalize(k, dim=-1).transpose(-2, -1) + rel_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, input_resolution, window_size=3, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, is_extrapolation=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if sr_ratio == 1:
            self.attn = Attention(
                dim,
                input_resolution,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                is_extrapolation=is_extrapolation)
        else:
            self.attn = AggregatedAttention(
                dim,
                input_resolution,
                window_size=window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                is_extrapolation=is_extrapolation)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale):
        x = x + self.drop_path(
            self.attn(self.norm1(x), H, W, relative_pos_index, relative_coords_table, seq_length_scale))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class TransNeXt(nn.Module):
    '''
    The parameter "img size" is primarily utilized for generating relative spatial coordinates,
    which are used to compute continuous relative positional biases. As this TransNeXt implementation can accept multi-scale inputs,
    it is recommended to set the "img size" parameter to a value close to the resolution of the inference images.
    It is not advisable to set the "img size" parameter to a value exceeding 800x800.
    The "pretrain size" refers to the "img size" used during the initial pre-training phase,
    which is used to scale the relative spatial coordinates for better extrapolation by the MLP.
    For models trained on ImageNet-1K at a resolution of 224x224,
    as well as downstream task models fine-tuned based on these pre-trained weights,
    the "pretrain size" parameter should be set to 224x224.
    '''

    def __init__(self, img_size=224, pretrain_size=None, window_size=[3, 3, 3, None],
                 patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, pretrained=None, scales=5,
                 is_extrapolation=False):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.window_size = window_size
        self.sr_ratios = sr_ratios
        self.is_extrapolation = is_extrapolation
        self.pretrain_size = pretrain_size or img_size
        self.scales = scales

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if not self.is_extrapolation:
                relative_pos_index, relative_coords_table = get_relative_position_cpb(
                    query_size=to_2tuple(img_size // (2 ** (i + 2))),
                    key_size=to_2tuple(img_size // ((2 ** (i + 2)) * sr_ratios[i])),
                    pretrain_size=to_2tuple(pretrain_size // (2 ** (i + 2))))

                self.register_buffer(f"relative_pos_index{i + 1}", relative_pos_index, persistent=False)
                self.register_buffer(f"relative_coords_table{i + 1}", relative_coords_table, persistent=False)

            patch_embed = OverlapPatchEmbed(patch_size=patch_size * 2 - 1 if i == 0 else 3,
                                            stride=patch_size if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], input_resolution=to_2tuple(img_size // (2 ** (i + 2))), window_size=window_size[i],
                num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], is_extrapolation=is_extrapolation)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        for n, m in self.named_modules():
            self._init_weights(m, n)
        if pretrained:
            self.init_weights(pretrained)

    def _init_weights(self, m: nn.Module, name: str = ''):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'query_embedding', 'relative_pos_bias_local', 'cpb', 'temperature'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            sr_ratio = self.sr_ratios[i]
            if self.is_extrapolation:
                relative_pos_index, relative_coords_table = get_relative_position_cpb(query_size=(H, W),
                                                                                      key_size=(
                                                                                          H // sr_ratio,
                                                                                          W // sr_ratio),
                                                                                      pretrain_size=to_2tuple(
                                                                                          self.pretrain_size // (
                                                                                                  2 ** (i + 2))),
                                                                                      device=x.device)
            else:
                relative_pos_index = getattr(self, f"relative_pos_index{i + 1}")
                relative_coords_table = getattr(self, f"relative_coords_table{i + 1}")

            with torch.no_grad():
                if i != (self.num_stages - 1):
                    local_seq_length = get_seqlen_scale((H, W), self.window_size[i], device=x.device)
                    seq_length_scale = torch.log(local_seq_length + (H // sr_ratio) * (W // sr_ratio))
                else:
                    seq_length_scale = torch.log(torch.as_tensor((H // sr_ratio) * (W // sr_ratio), device=x.device))

            for blk in block:
                x = blk(x, H, W, relative_pos_index, relative_coords_table, seq_length_scale)

            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            if i >= (5 - self.scales):
                outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


@MODELS.register_module()
class transnext_tiny(TransNeXt):
    def __init__(self, **kwargs):
        super().__init__(window_size=[3, 3, 3, None],
                         patch_size=4, embed_dims=[72, 144, 288, 576], num_heads=[3, 6, 12, 24],
                         mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 15, 2], sr_ratios=[8, 4, 2, 1],
                         drop_rate=0.0, drop_path_rate=0.3,
                         pretrained=kwargs['pretrained'], img_size=kwargs['img_size'],
                         pretrain_size=kwargs['pretrain_size'], scales=kwargs['scales'])


@MODELS.register_module()
class transnext_small(TransNeXt):
    def __init__(self, **kwargs):
        super().__init__(window_size=[3, 3, 3, None],
                         patch_size=4, embed_dims=[72, 144, 288, 576], num_heads=[3, 6, 12, 24],
                         mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[5, 5, 22, 5], sr_ratios=[8, 4, 2, 1],
                         drop_rate=0.0, drop_path_rate=0.5,
                         pretrained=kwargs['pretrained'], img_size=kwargs['img_size'],
                         pretrain_size=kwargs['pretrain_size'], scales=kwargs['scales'])


@MODELS.register_module()
class transnext_base(TransNeXt):
    def __init__(self, **kwargs):
        super().__init__(window_size=[3, 3, 3, None],
                         patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[4, 8, 16, 32],
                         mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[5, 5, 23, 5], sr_ratios=[8, 4, 2, 1],
                         drop_rate=0.0, drop_path_rate=0.6,
                         pretrained=kwargs['pretrained'], img_size=kwargs['img_size'],
                         pretrain_size=kwargs['pretrain_size'], scales=kwargs['scales'])
