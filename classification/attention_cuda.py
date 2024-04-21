import torch
import torch.nn as nn
import torch.nn.functional as F
import swattention
import numpy as np

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


@torch.no_grad()
def get_seqlen_scale(input_resolution, window_size):
    return torch.nn.functional.avg_pool2d(torch.ones(1, input_resolution[0], input_resolution[1]) * (window_size ** 2),
                                          window_size, stride=1, padding=window_size // 2, ).reshape(-1, 1)


class AggregatedAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1, fixed_pool_size=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.sr_ratio = sr_ratio

        assert window_size % 2 == 1, "window size must be odd"
        self.window_size = window_size
        self.local_len = window_size ** 2

        if fixed_pool_size is None:
            self.pool_H, self.pool_W = input_resolution[0] // self.sr_ratio, input_resolution[1] // self.sr_ratio
        else:
            assert fixed_pool_size < min(input_resolution), \
                f"The fixed_pool_size {fixed_pool_size} should be less than the shorter side of input resolution {input_resolution} to ensure pooling works correctly."
            self.pool_H, self.pool_W = fixed_pool_size, fixed_pool_size
        self.pool_len = self.pool_H * self.pool_W

        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Components to generate pooled features.
        self.pool = nn.AdaptiveAvgPool2d((self.pool_H, self.pool_W))
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        # relative bias for local features
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0, std=0.0004))

        # Generate padding_mask && sequnce length scale
        local_seq_length = get_seqlen_scale(input_resolution, window_size)
        self.register_buffer("seq_length_scale", torch.as_tensor(np.log(local_seq_length.numpy() + self.pool_len)),
                             persistent=False)

        # dynamic_local_bias:
        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
        B, N, C = x.shape

        # Generate queries, normalize them with L2, add query embedding, and then magnify with sequence length scale and temperature.
        # Use softplus function ensuring that the temperature is not lower than 0.
        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * self.seq_length_scale

        # Generate unfolded keys and values and l2-normalize them
        k_local, v_local = self.kv(x).reshape(B, N, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3).chunk(2,
                                                                                                                 dim=1)

        # Compute local similarity
        attn_local = sw_qkrpb_cuda.apply(q_norm_scaled.contiguous(), F.normalize(k_local, dim=-1).contiguous(),
                                         self.relative_pos_bias_local,
                                         H, W, self.window_size)

        # Generate pooled features
        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = self.pool(self.act(self.sr(x_))).reshape(B, -1, self.pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        # Generate pooled keys and values
        kv_pool = self.kv(x_).reshape(B, self.pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        # Use MLP to generate continuous relative positional bias for pooled features.
        pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                    relative_pos_index.view(-1)].view(-1, N, self.pool_len)
        # Compute pooled similarity
        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias

        # Concatenate local & pooled similarity matrices and calculate attention weights through the same Softmax
        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Split the attention weights and separately aggregate the values of local & pooled features
        attn_local, attn_pool = torch.split(attn, [self.local_len, self.pool_len], dim=-1)
        attn_local = (q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local
        x_local = sw_av_cuda.apply(attn_local.type_as(v_local), v_local.contiguous(), H, W, self.window_size)

        x_pool = attn_pool @ v_pool
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        # Linear projection and output
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
