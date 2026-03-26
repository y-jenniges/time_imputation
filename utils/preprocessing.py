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
