def get_lr_scales(n_groups, lr_decay):
    res = [1] * n_groups
    for i in range(1, n_groups):
        res[i] = res[i - 1] * lr_decay
    return res[::-1]


def get_param_groups(model, lr_decay=None, weight_decay=0):
    if not lr_decay and not weight_decay:
        return model.parameters()
    
    no_weight_decay = set()
    if hasattr(model, 'no_weight_decay'):
        no_weight_decay = model.no_weight_decay

    param_groups = {}
    
    if lr_decay:
        assert hasattr(model, "num_lr_scale_groups") and hasattr(model, "lr_scale_group_id"),\
            "Layer-wise lr decay is not supported by the given model."
        lr_scales = get_lr_scales(model.num_lr_scale_groups, lr_decay)

    for name, param in model.named_parameters():
        if not param.requires_grad: continue

        options = []
        
        if lr_decay:
            layer_id = model.lr_scale_group_id(name)
            options.append(f"lrd{layer_id}")
        
        # Weight decay only matmuls & embeddings.
        if weight_decay and param.dim() >= 2 and name not in no_weight_decay:
            wd = weight_decay
            options.append("wd")
        else:
            wd = 0
            options.append("no_wd")
        
        if not options: continue
        group_name = '_'.join(options)
        
        # Init param group.
        if group_name not in param_groups:
            param_group = {"params": [], "weight_decay": wd}
            if lr_decay: param_group["lr_scale"] = lr_scales[layer_id]
            param_groups[group_name] = param_group
        
        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())