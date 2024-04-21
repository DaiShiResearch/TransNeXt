# --------------------------------------------------------
# TransNeXt
# Code modification based on Swin Transformer, Focal Transformer, and timm.optim
# To support setting no weight decay based on parameter name keywords
# Modified by Dai Shi (daishiresearch@gmail.com)
# --------------------------------------------------------

from torch import optim as optim

def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        opt=cfg.opt,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum)
    if getattr(cfg, 'opt_eps', None) is not None:
        kwargs['eps'] = cfg.opt_eps
    if getattr(cfg, 'opt_betas', None) is not None:
        kwargs['betas'] = cfg.opt_betas
    if getattr(cfg, 'opt_args', None) is not None:
        kwargs.update(cfg.opt_args)
    return kwargs

def build_optimizer(args, model):
    """ Legacy optimizer factory for backwards compatibility.
    NOTE: Use create_optimizer_v2 for new code.
    """
    return build_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args)
    )


def build_optimizer_v2(model,
                    opt: str = 'sgd',
                    lr = None,
                    weight_decay: float = 0.,
                    momentum: float = 0.9,
                    **kwargs):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_args = dict(weight_decay=weight_decay, **kwargs)

    opt_lower = opt.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=momentum, lr=lr, nesterov=True, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, lr=lr, **opt_args)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
