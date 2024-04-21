_base_ = ['./mask2former_r50_8xb2-160k_ade20k-512x512.py']
depths = [5, 5, 23, 5]
model = dict(
    backbone=dict(
        type='transnext_base',
        pretrain_size=224,
        img_size=512,
        pretrained=None),
    decode_head=dict(in_channels=[96, 192, 384, 768]))


backbone_nodecay = dict(lr_mult=0.1, decay_mult=0)
backbone_decay = dict(lr_mult=0.1)
embed_mult = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {'attn.query_embedding': backbone_nodecay,
               'relative_pos_bias_local': backbone_nodecay,
               'cpb': backbone_nodecay,
               'temperature': backbone_nodecay,
               'attn.learnable': backbone_decay,
               'attn.q.weight': backbone_decay,
               'attn.q.bias': backbone_nodecay,
               'attn.kv.weight': backbone_decay,
               'attn.kv.bias': backbone_nodecay,
               'attn.qkv.weight': backbone_decay,
               'attn.qkv.bias': backbone_nodecay,
               'attn.sr.weight': backbone_decay,
               'attn.sr.bias': backbone_nodecay,
               'attn.norm': backbone_nodecay,
               'attn.proj.weight': backbone_decay,
               'attn.proj.bias': backbone_nodecay,
               'mlp.fc1.weight': backbone_decay,
               'mlp.fc2.weight': backbone_decay,
               'mlp.fc1.bias': backbone_nodecay,
               'mlp.fc2.bias': backbone_nodecay,
               'mlp.dwconv.dwconv.weight': backbone_decay,
               'mlp.dwconv.dwconv.bias': backbone_nodecay,
               'decode_head.query_embed': embed_mult,
               'decode_head.query_feat': embed_mult,
               'decode_head.level_embed': embed_mult
               }
custom_keys.update({
    f'backbone.norm{stage_id + 1}': backbone_nodecay
    for stage_id in range(len(depths) - 1)
})
custom_keys.update({
    f'backbone.patch_embed{stage_id + 1}.norm': backbone_nodecay
    for stage_id in range(len(depths) - 1)
})
custom_keys.update({
    f'backbone.patch_embed{stage_id + 1}.proj.weight': backbone_decay
    for stage_id in range(len(depths) - 1)
})
custom_keys.update({
    f'backbone.patch_embed{stage_id + 1}.proj.bias': backbone_nodecay
    for stage_id in range(len(depths) - 1)
})
custom_keys.update({
    f'backbone.block{stage_id + 1}.{block_id}.norm': backbone_nodecay
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, bias_decay_mult=0, norm_decay_mult=0, flat_decay_mult=0),
    #accumulative_counts=2,
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa
# train_dataloader = dict(
#     batch_size=1, )
