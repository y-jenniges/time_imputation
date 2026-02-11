import torch
import torch.nn as nn
import torch.nn.functional as F
import gsw


def build_loss(loss_config):
    loss_class = loss_config["class"]
    kwargs = loss_config.get("kwargs", {})
    return loss_class(**kwargs)


def name_to_loss_spec(loss_name):
    loss_spec = {}
    if loss_name == "mse":
        loss_spec["class"] = MaskedMSELoss
        loss_spec["kwargs"] = {}
    elif loss_name == "hetero":
        loss_spec["class"] = HeteroscedasticLoss
        loss_spec["kwargs"]  = {}
    return loss_spec


class BaseLoss(nn.Module):
    def forward(self, input, target, mask=None, coords=None, pred_var=None, **kwargs):
        """ Computes loss.
        Args:
            input (torch.Tensor): Model prediction (mean)
            target (torch.Tensor): Ground truth
            mask (torch.BoolTensor, optional): Mask  of values to consider
            coords (torch.Tensor, optional): Coordinates (for PhysicsLoss)
            pred_var (torch.Tensor, optional): Prediction variance (for HeteroscedasticLoss)
            kwargs: Extra optional arguments

        Returns:
            torch.Tensor: Scalar loss
        """
        raise NotImplementedError

# @todo optionally use n_loss_mask to evaluate error on query point AND neighbours
class MaskedMSELoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, mask=None, coords=None, pred_var=None, **kwargs):
        # Only use not-nan entries (from ground truth)
        valid_mask = mask & (~torch.isnan(target))
        if valid_mask.sum() == 0:
            return None

        mse = F.mse_loss(input[valid_mask], target[valid_mask], reduction="mean")
        return mse


class HeteroscedasticLoss(BaseLoss):
    def __init__(self):  # , eps: float = 1e-8):
        super().__init__()

    def forward(self, input, target, mask=None, coords=None, pred_var=None, **kwargs):
        # Only use not-nan entries (from ground truth)
        valid_mask = mask & (~torch.isnan(target))
        if valid_mask.sum() == 0:
            return None

        sq_error = (input[valid_mask] - target[valid_mask]) ** 2
        loss = 0.5 * (sq_error / (pred_var[valid_mask]) + torch.log(pred_var[valid_mask]))
        return loss.mean()


class PhysicsLoss(BaseLoss):
    def __init__(self, base_loss: BaseLoss = MaskedMSELoss(), lambda_phys: float = 0.1, bounds=None, spacings=None):
        super().__init__()
        self.base_loss = base_loss
        self.lambda_phys = lambda_phys
        self.bounds = bounds
        self.spacings = spacings

    def forward(self, input, target, mask=None, coords=None, pred_var=None, **kwargs):
        # Only use not-nan entries (from ground truth)
        valid_mask = mask & (~torch.isnan(target))
        if valid_mask.sum() == 0:
            return None

        # Compute base loss
        base_loss_kwargs = {"input": input, "target": target, "coords": coords, "pred_var": pred_var, "mask": mask}
        base_loss_val = self.base_loss(**base_loss_kwargs)

        phys_loss_val = 0
        # Spatiotemporal smoothness penalty (finite differences)
        if coords is not None and self.spacings is not None:
            batch_size, set_size, feature_size = input.shape
            coords_size = coords.shape[2]

            # Flatten
            input_flat = input.reshape(batch_size*set_size, feature_size)
            coords_flat = coords.reshape(batch_size*set_size, coords_size)

            for dim in range(coords_size):
                # Sort values and coords
                sorted_idx = torch.argsort(coords_flat[:, dim])
                sorted_input = input_flat[sorted_idx]
                sorted_coord = coords_flat[:, dim][sorted_idx]

                # Compute gradient and loss for this dimension
                grad = torch.gradient(sorted_input, spacing=sorted_coord, edge_order=2)[0]
                phys_loss_val += torch.mean(grad ** 2)

        # Enforce physical bounds
        if self.bounds is not None:
            # Flatten
            batch_size, set_size, feature_size = input.shape
            input_flat = input.reshape(batch_size * set_size, feature_size)

            y_min, y_max = self.bounds
            lower = torch.clamp(y_min - input_flat, min=0) ** 2
            upper = torch.clamp(input_flat - y_max, min=0) ** 2
            phys_loss_val += lower.mean() + upper.mean()

        return (1 - self.lambda_phys) * base_loss_val + self.lambda_phys * phys_loss_val


def smoothness_loss(pred, coords, scale=1.0):
    """
    Penalize large gradients in prediction w.r.t. spatial/temporal coords.

    pred: [N, value_dim] tensor (reconstructed field)
    coords: [N, coord_dim] tensor (lon, lat, depth, time)
    scale: float, controls strength of smoothness penalty
    """
    # approximate local gradients with finite differences on neighboring points
    # assume coords are normalized roughly to [0,1]
    diff_pred = pred[1:] - pred[:-1]
    diff_coords = coords[1:] - coords[:-1]
    spatial_distance = (diff_coords ** 2).sum(dim=1, keepdim=True).sqrt() + 1e-8

    gradient = (diff_pred ** 2).sum(dim=1, keepdim=True) / spatial_distance
    smoothness_loss = gradient.mean()

    return scale * smoothness_loss


def physics_loss(y_true, pred_mean, pred_var, mask, scalers=None, cfg=None):
    # Example signature:
    # y_true, pred_mean, pred_var: tensors shape (B, C) or (B, H, W, D, C)
    # mask: same shape but 0/1 (float) indicating supervised positions
    # scalers: dict of sklearn-like scalers with inverse_transform (for 'T','S','depth' keys)
    # cfg: dict with lambda weights and small eps
    if cfg is None:
        cfg = {
            "eps": 1e-8,
            "lambda_vert": 1e-3,
            "lambda_density": 1e-2,
            "lambda_bounds": 1e-3,
            "lambda_time": 1e-3,
            "bounds": {},  # e.g. {'T':(-2.5,35), 'S':(0,42)}
            "dz": 1.0,  # vertical spacing in meters (if uniform)
            "use_density": True
        }

    eps = cfg["eps"]

    # --- heteroscedastic NLL using variance
    s = pred_var
    sq_error = (y_true - pred_mean) ** 2
    # elementwise NLL (0.5 * (exp(-s) * sq_error + s))
    nll_elem = 0.5 * (torch.exp(-s) * sq_error + s)
    recon_loss = (nll_elem * mask).sum() / (mask.sum() + eps)

    total_loss = recon_loss
    components = {"recon": recon_loss.detach()}

    # --- vertical gradient penalty (apply on mean predictions only) ---
    if cfg["lambda_vert"] > 0:
        # assumes data layout [..., depth, C] or [..., C] where depth dimension known.
        # Here we assume shape (B, D, C) or (B, H, W, D, C). User may adapt slicing.
        # We'll implement a generic depth-diff for 3D shape (B, D, C)
        # User must reshape accordingly for their tensors.
        try:
            # Attempt depth-last 3D: (B, D, C)
            pred_m = pred_mean
            # compute forward difference along depth axis (assumed axis=1)
            grad_z = pred_m[:, 1:, :] - pred_m[:, :-1, :]
            # mask for gradient (exclude top/bottom where no neighbor)
            mask_grad = mask[:, 1:, :] * mask[:, :-1, :]
            vert_pen = ((grad_z ** 2) * mask_grad).sum() / (mask_grad.sum() + eps)
        except Exception:
            vert_pen = torch.tensor(0.0, device=pred_mean.device)

        total_loss = total_loss + cfg["lambda_vert"] * vert_pen
        components["vert_pen"] = vert_pen.detach()

    # --- temporal consistency penalty ---
    if cfg["lambda_time"] > 0:
        # assumes time dimension is axis=1 for shape (B, T, C) or adapt accordingly
        try:
            pred_m = pred_mean
            dt = pred_m[:, 1:, :] - pred_m[:, :-1, :]
            mask_t = mask[:, 1:, :] * mask[:, :-1, :]
            time_pen = ((dt ** 2) * mask_t).sum() / (mask_t.sum() + eps)
        except Exception:
            time_pen = torch.tensor(0.0, device=pred_mean.device)

        total_loss = total_loss + cfg["lambda_time"] * time_pen
        components["time_pen"] = time_pen.detach()

    # --- bounds penalty: keep predictions within physically plausible min/max ---
    if cfg["lambda_bounds"] > 0 and "bounds" in cfg:
        bpen = 0.0
        cnt = 0
        for var_idx, var_name in enumerate(cfg.get("var_names", [])):
            if var_name in cfg["bounds"]:
                lo, hi = cfg["bounds"][var_name]
                pred_v = pred_mean[..., var_idx]
                # penalize squared exceedance
                under = F.relu(lo - pred_v)
                over = F.relu(pred_v - hi)
                mask_v = mask[..., var_idx]
                pen = ((under + over) ** 2 * mask_v).sum() / (mask_v.sum() + eps)
                bpen = bpen + pen
                cnt += 1
        if cnt > 0:
            bpen = bpen / cnt
        else:
            bpen = torch.tensor(0.0, device=pred_mean.device)

        total_loss = total_loss + cfg["lambda_bounds"] * bpen
        components["bounds_pen"] = bpen.detach()

    # --- density stability penalty (requires inverse-scaling to physical units) ---
    # Only compute if scalers provided and lambda_density>0
    if cfg.get("lambda_density", 0) > 0 and cfg.get("use_density", True):
        try:
            # assume T var index and S var index known in cfg
            T_idx = cfg.get("T_idx", 0)
            S_idx = cfg.get("S_idx", 1)
            depth_idx = cfg.get("depth_idx", None)  # if depth is variable input
            # inverse transform only masked positions to physical units
            # flatten for inverse transform then reshape back
            device = pred_mean.device
            mask_phys = mask[..., T_idx] > 0  # boolean
            if mask_phys.any():
                # collect predicted T,S at masked supervised positions
                pred_T = pred_mean[..., T_idx].detach().cpu().numpy()
                pred_S = pred_mean[..., S_idx].detach().cpu().numpy()
                # NOTE: user must implement correct inverse_transform function for their scalers.
                T_phys = scalers['T'].inverse_transform(pred_T.reshape(-1, 1)).reshape(pred_T.shape)
                S_phys = scalers['S'].inverse_transform(pred_S.reshape(-1, 1)).reshape(pred_S.shape)
                # depth (pressure) needed - if depth is a channel, inverse it too (or use known grid)
                if depth_idx is not None:
                    depth_pred = pred_mean[..., depth_idx].detach().cpu().numpy()
                    z_phys = scalers['depth'].inverse_transform(depth_pred.reshape(-1, 1)).reshape(depth_pred.shape)
                else:
                    # if depth grid known, provide z array in cfg
                    z_phys = cfg.get('z_grid')  # shape must broadcast
                # compute density via TEOS-10 (gsw), requires arrays in numpy
                # WARNING: require `gsw` module installed
                import gsw
                rho = gsw.rho(S_phys, T_phys, z_phys)
                # compute vertical density gradient along depth axis (axis=1 typically)
                # reshape back as torch and compute negative gradient penalty (ReLU of -drho)
                rho_torch = torch.tensor(rho, device=device, dtype=pred_mean.dtype)
                drho = rho_torch[:, 1:, ...] - rho_torch[:, :-1, ...]
                inv_stab = F.relu(-drho)  # positive values when unstable
                # average only where both levels exist (masking!)
                # create mask for both levels from original mask if available
                # For simplicity average over all drho entries
                density_pen = inv_stab.mean()
            else:
                density_pen = torch.tensor(0.0, device=pred_mean.device)
        except Exception as e:
            # if gsw not available or inverse transform fails, skip gracefully
            density_pen = torch.tensor(0.0, device=pred_mean.device)

        total_loss = total_loss + cfg["lambda_density"] * density_pen
        components["density_pen"] = density_pen.detach()

    return total_loss, components

