_base_ = './dino-4scale_r50_8xb2-12e_coco.py'

fp16 = dict(loss_scale=512.)
depths = [5, 5, 22, 5]
# optimizer
num_levels = 5
model = dict(
    num_feature_levels=num_levels,
    backbone=dict(
        type='transnext_small',
        pretrain_size=224,
        img_size=800,
        pretrained=None,
        scales=num_levels),
    neck=dict(in_channels=[72, 144, 288, 576], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))))
backbone_nodecay = dict(lr_mult=0.1, decay_mult=0)
backbone_decay = dict(lr_mult=0.1)
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
               'mlp.dwconv.dwconv.bias': backbone_nodecay, }
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
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        betas=(0.9, 0.999), weight_decay=0.05, ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys=custom_keys, bias_decay_mult=0, norm_decay_mult=0, flat_decay_mult=0),
    accumulative_counts=2,
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa
train_dataloader = dict(
    batch_size=1, )

