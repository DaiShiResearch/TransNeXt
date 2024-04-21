cfg = dict(
    model='transnext_small',
    pretrain_size=224,
    input_size=384,
    drop_path=0.7,
    lr=1e-5,
    clip_grad=1.0,
    epochs=5,
    cutmix=0,
    sched=None,
    weight_decay=0.05,
    output_dir='checkpoints/transnext_small_384',
)
