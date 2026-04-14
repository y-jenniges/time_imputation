import torch


def fill_feature_tensor(features, mask, fill_strategy="zero", mean_values=None):
    # If no mask is provided, infer from NaNs
    if mask is None:
        mask = ~torch.isnan(features)

    if fill_strategy == "zero":
        return torch.where(mask, features, torch.zeros_like(features))
    elif fill_strategy == "mean":
        return torch.where(mask, features, torch.zeros_like(features) if mean_values is None else mean_values.unsqueeze(0))
    else:
        raise ValueError(f"Unknown fill_strategy: {fill_strategy}")


def get_scopes(cfg):
    if cfg.attention_type == "space_time_attention":
        return ["space", "time"]
    elif cfg.attention_type == "time_space_attention":
        return ["time", "space"]
    elif cfg.attention_type == "space_time_depth_attention" or cfg.attention_type == "weighted_space_time_depth_attention":
        return ["space", "time", "depth"]
    elif cfg.graph_mode == "time_sequence":
        return ["time"]
    else:
        return ["default"]
