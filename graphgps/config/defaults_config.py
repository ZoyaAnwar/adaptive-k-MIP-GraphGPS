from torch_geometric.graphgym.register import register_config


@register_config('overwrite_defaults')
def overwrite_defaults_cfg(cfg):
    """Overwrite the default config values that are first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg

    WARNING: At the time of writing, the order in which custom config-setting
    functions like this one are executed is random; see the referenced `set_cfg`
    Therefore never reset here config options that are custom added, only change
    those that exist in core GraphGym.
    """

    # Overwrite default dataset name
    cfg.dataset.name = 'none'

    # Overwrite default rounding precision
    cfg.round = 5


@register_config('extended_cfg')
def extended_cfg(cfg):
    """General extended config options.
    """

    # Additional name tag used in `run_dir` and `wandb_name` auto generation.
    cfg.name_tag = ""

    # Torch device string used throughout training/logging. Upstream GraphGym
    # replaced this with `accelerator`; keep `device` for this codebase.
    cfg.device = 'cpu'

    # Training loop mode: 'standard' (GraphGym) or a registered custom loop
    # name such as 'custom' (see graphgps.train). Removed from upstream
    # GraphGym; keep it here for this codebase.
    cfg.train.mode = 'custom'

    # In training, if True (and also cfg.train.enable_ckpt is True) then
    # always checkpoint the current best model based on validation performance,
    # instead, when False, follow cfg.train.eval_period checkpointing frequency.
    cfg.train.ckpt_best = False

    cfg.train.shuffle = True

    cfg.share.gradient_checkpointing = False

    cfg.model.auto_satisfy_param_budget = False
    cfg.model.param_budget = 500000

    cfg.train.eval_smoothing_metrics = False