"""Loss functions for MaSTNeT training: masked MSE, heteroscedastic NLL, physics and Fourier losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_loss(loss_config):
    """Instantiate a loss class from a ``{'class': ..., 'kwargs': {...}}`` specification dict."""
    loss_class = loss_config["class"]
    kwargs = loss_config.get("kwargs", {})
    return loss_class(**kwargs)


def name_to_loss_spec(loss_name):
    """
    Map a loss name string to a loss class and keyword arguments.

    Parameters
    ----------
    loss_name : str
        One of ``'mse'``, ``'hetero'``, ``'physics_hetero'``, ``'hetero_smooth'``,
        ``'hetero_fourier'``.

    Returns
    -------
    dict
        Specification dict with keys ``'class'`` and ``'kwargs'``.
    """
    loss_spec = {}
    if loss_name == "mse":
        loss_spec["class"] = MaskedMSELoss
        loss_spec["kwargs"] = {}
    elif loss_name == "hetero":
        loss_spec["class"] = HeteroscedasticGaussianNLL
        loss_spec["kwargs"] = {}
    elif loss_name == "physics_hetero":
        loss_spec["class"] = PhysicsLoss
        loss_spec["kwargs"] = {"sigma": 1.0, "lambda_smooth": 1e-3}
    elif loss_name == "hetero_smooth":
        loss_spec["class"] = PhysicsLoss
        loss_spec["kwargs"] = {"base_loss": HeteroscedasticGaussianNLL(), "sigma": 1.0, "lambda_smooth": 1e-3}
    elif loss_name == "hetero_fourier":
        loss_spec["class"] = HeteroscedasticNLLWithFourierSparsity
        loss_spec["kwargs"] = {"lambda_fil": 0.01, "spatial_dims": None, "temporal_dim": None}
    else:
        raise ValueError(f"Unknown loss name {loss_name}")
    return loss_spec


class BaseLoss(nn.Module):
    """Abstract base class for all loss functions; subclasses must implement ``forward``."""

    def forward(self, input, target, mask=None, coords=None, pred_var=None, **kwargs):
        """
        Compute loss on (optionally masked) predictions vs. targets.

        Parameters
        ----------
        input : Tensor
            Model predictions.
        target : Tensor
            Ground truth; ``NaN`` at missing positions.
        mask : Tensor or None
            Boolean mask - True at positions to include in the loss.
        coords : Tensor or None
            Spatiotemporal coordinates; required by :class:`PhysicsLoss`.
        pred_var : Tensor or None
            Predicted log-variance; required by heteroscedastic losses.
        **kwargs
            Accepted for API compatibility.

        Returns
        -------
        Tensor or None
            Scalar loss, or None if no valid positions exist.
        """
        raise NotImplementedError


class MaskedMSELoss(BaseLoss):
    """MSE loss computed only on observed (mask-selected) positions."""

    def __init__(self):
        super().__init__()

    def forward(self, input, target, mask=None, coords=None, pred_var=None, **kwargs):
        # Only use not-nan entries (from ground truth)
        if mask is None:
            valid_mask = ~torch.isnan(target)
        else:
            valid_mask = mask & (~torch.isnan(target))

        if valid_mask.sum() == 0:
            return None

        valid_mask = valid_mask.to(input.device)

        mse = F.mse_loss(input[valid_mask], target[valid_mask], reduction="mean")
        return mse


class HeteroscedasticGaussianNLL(BaseLoss):
    """
    Gaussian NLL with predicted log-variance for numeric stability (Kendall & Gal 2017).

    Reference:
    Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
    In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), Advances in
    Neural Information Processing Systems (Vol. 30). Curran Associates, Inc.
    https://proceedings.neurips.cc/paper_files/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, target, mask=None, coords=None, pred_var=None, **kwargs):
        # Only use not-nan entries (from ground truth)
        if mask is None:
            valid_mask = ~torch.isnan(target)
        else:
            valid_mask = mask & (~torch.isnan(target))

        if valid_mask.sum() == 0:
            return None

        sq_error = (input[valid_mask] - target[valid_mask]) ** 2
        loss = 0.5 * (sq_error * (torch.exp(-1 * pred_var[valid_mask])) + pred_var[valid_mask])
        return loss.mean()


class PhysicsLoss(BaseLoss):
    """
    Base loss augmented with physics-inspired penalties (spatio-temporal smoothness and value bounds).

    Parameters
    ----------
    base_loss : BaseLoss, default=MaskedMSELoss()
        Pointwise reconstruction loss applied before the penalty terms.
    sigma : float, default=1.0
        Length-scale for the RBF smoothness kernel.
    lambda_smooth : float, default=0.0
        Weight for the spatiotemporal smoothness penalty.
    lambda_bounds : float, default=0.0
        Weight for the physical bounds penalty.
    bounds : tuple or None, default=None
        ``(y_min, y_max)`` per-feature bounds tensors; required when ``lambda_bounds > 0``.
    """

    def __init__(self, base_loss: BaseLoss = MaskedMSELoss(), sigma=1.0, lambda_smooth: float = 0.0,
                 lambda_bounds: float = 0.0, bounds=None):
        super().__init__()
        self.base_loss = base_loss
        self.lambda_smooth = lambda_smooth
        self.lambda_bounds = lambda_bounds
        self.bounds = bounds
        self.sigma = sigma

    def forward(self, input, target, mask=None, pred_var=None,
                coords=None, neighbour_features=None, neighbour_coords=None,
                query_coords=None,
                neighbours=None,
                anisotropic_weights=None,
                **kwargs):
        # Only use not-nan entries (from ground truth)
        if mask is None:
            valid_mask = ~torch.isnan(target)
        else:
            valid_mask = mask & (~torch.isnan(target))

        if valid_mask.sum() == 0:
            return None

        # Compute base loss
        base_loss_kwargs = {"input": input, "target": target, "coords": coords, "pred_var": pred_var, "mask": mask}
        base_loss_val = self.base_loss(**base_loss_kwargs)

        # Spatiotemporal smoothness penalty
        smooth_loss_val = 0.0
        if neighbours is not None:  # and neighbour_coords is not None:
            smooth_loss_val += self._compute_smoothness_loss(input=input,
                                                             neighbour_features=neighbours["default"]["features"],
                                                             # @todo what to do with multiple scopes? smoothness in every direction?
                                                             neighbour_coords=neighbours["default"]["coords"],
                                                             query_coords=query_coords,
                                                             anisotropic_weights=anisotropic_weights)

        # Enforce physical bounds
        bounds_loss_val = 0.0
        if self.bounds is not None:
            bounds_loss_val += self._compute_bounds_loss(input=input, mask=valid_mask)

        return base_loss_val + self.lambda_smooth * smooth_loss_val + self.lambda_bounds * bounds_loss_val

    def _compute_smoothness_loss(self, input, neighbour_features, neighbour_coords, query_coords,
                                 anisotropic_weights=None):
        """ Distance-weighted smoothness penalty. """
        valid = ~torch.isnan(neighbour_features)
        n_feat = torch.nan_to_num(neighbour_features)

        # Squared spatio-temporal distances
        dist2_coords = ((query_coords.unsqueeze(1) - neighbour_coords) ** 2)  # [B, K, C]

        if anisotropic_weights is not None:
            # Ensure positivity
            pos_aniso_weights = torch.exp(anisotropic_weights)

            # Broadcast: [1,1,C]
            dist2_coords = (dist2_coords * pos_aniso_weights.view(1, 1, -1))  # [B, K]

        # RBF kernel edge weight
        w = torch.exp(-dist2_coords.sum(dim=-1) / (2 * self.sigma ** 2)).unsqueeze(-1)  # [B, K, 1]

        # Squared feature difference
        diff2 = (input.unsqueeze(1) - n_feat) ** 2  # [B, K, F]
        weighted = w * diff2 * valid
        denom = (w * valid).sum()

        if valid.sum() == 0:
            return torch.tensor(0.0, device=input.device)

        return weighted.sum() / denom

    def _compute_bounds_loss(self, input, mask):
        """ Feature bound penalty. """
        y_min, y_max = self.bounds

        lower = torch.clamp(y_min - input, min=0) ** 2
        upper = torch.clamp(input - y_max, min=0) ** 2

        # Only consider valid entries
        lower = lower * mask.float()
        upper = upper * mask.float()

        loss = (lower.mean() + upper.mean()) / mask.sum()
        return loss


class FourierImputationLoss(BaseLoss):
    """
    Fourier Imputation Loss (FIL) (Nie et al., KDD 2024).

    Encourages the imputed field to have a sparse Fourier spectrum, equivalent to
    minimising the convolution nuclear norm (low-rank regularisation in the frequency
    domain). Applied to the full composite field (observed GT + imputed predictions),
    so gradients flow only through the imputed positions.

    For ocean data this is well-motivated: physical fields (SST, salinity) are dominated
    by large-scale modes with most energy at low spatial and temporal frequencies.

    Reference:
    Nie, T., Qin, G., Ma, W., Mei, Y., & Sun, J. (2024). ImputeFormer: Low Rankness-Induced Transformers for
    Generalizable Spatiotemporal Imputation. Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and
    Data Mining, 2260–2271. https://www.doi.org/10.1145/3637528.3671751

    Parameters
    ----------
    spatial_dims : list[int] or None, default=None
        Dimensions corresponding to space in the input tensor. If None, defaults to all
        middle dims (excluding batch dim 0 and feature dim -1).
        Example: ``[1, 2]`` for a ``[B, H, W, T, F]`` tensor.
    temporal_dim : int or None, default=None
        Dimension corresponding to time. If None, FIL is applied over spatial dims only
        (or all non-feature dims when ``spatial_dims`` is also None).
    """

    def __init__(self, spatial_dims=None, temporal_dim=None):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.temporal_dim = temporal_dim

    def forward(self, input, target, mask=None, pred_var=None, **kwargs):
        """
        Parameters
        ----------
        input : Tensor, shape [B, ..., F]
            Predicted means (any spatiotemporal layout).
        target : Tensor, shape [B, ..., F]
            Ground truth with NaNs at missing locations.
        mask : Tensor or None
            Boolean - True where values are observed. Derived from ``~isnan(target)``
            when None.
        pred_var : ignored
            Accepted for API consistency with other loss classes.
        """
        # --- Build the composite field: observed GT + imputed predictions ----------
        # We want to apply the FFT over the full field. At observed locations we use
        # the ground truth (no gradient needed there), at missing locations we use
        # the model's prediction (gradient flows through here).
        # This matches Eq. (14) in the ImputeFormer paper exactly.
        if mask is None:
            obs_mask = ~torch.isnan(target)
        else:
            obs_mask = mask & (~torch.isnan(target))

        # Fill NaNs in target with model predictions so the composite is NaN-free
        target_filled = torch.where(obs_mask, target, input)

        # Composite: GT where observed, prediction where missing
        # Use detach() on the GT part - we only want gradients through the imputed part
        composite = torch.where(
            obs_mask,
            target_filled.detach(),  # no gradient through known observations
            input  # gradient flows through imputed values
        )

        # --- Determine which dims to transform ---------------------------------
        # We exclude the batch dim (0) and the feature dim (-1) by default.
        n_dims = composite.ndim
        if self.spatial_dims is not None or self.temporal_dim is not None:
            fft_dims = []
            if self.spatial_dims is not None:
                fft_dims.extend(self.spatial_dims)
            if self.temporal_dim is not None:
                fft_dims.append(self.temporal_dim)
        else:
            # Default: all middle dims (exclude batch and feature)
            fft_dims = list(range(1, n_dims - 1))

        # --- Compute FFT and l1 norm -------------------------------------------
        # torch.fft.fftn handles multi-dim FFT natively.
        # We take the magnitude (absolute value) before the ℓ1 norm.
        spectrum = torch.fft.fftn(composite, dim=fft_dims)
        magnitude = torch.abs(spectrum)  # [B, ..., F]

        # l1 norm averaged over all elements (consistent with paper's 1/NT normalisation)
        fil_loss = magnitude.mean()

        return fil_loss


class HeteroscedasticNLLWithFourierSparsity(BaseLoss):
    """
    Combined loss: Heteroscedastic Gaussian NLL + Fourier Imputation Loss (FIL).

        L = NLL(μ, log_σ², y) + λ_fil · FIL(μ, y)

    The NLL term handles pointwise accuracy and uncertainty calibration; the FIL term
    regularises the global spectral structure of the imputed field, encouraging
    physically coherent, low-rank solutions.

    Parameters
    ----------
    lambda_fil : float, default=0.01
        Weight for the Fourier sparsity regulariser. Start small (1e-3 to 1e-2) and
        tune; too large --> over-smoothed means; too small --> no spectral benefit.
    spatial_dims : list[int] or None, default=None
        Spatial dimensions for FFT (see :class:`FourierImputationLoss`).
    temporal_dim : int or None, default=None
        Temporal dimension for FFT (see :class:`FourierImputationLoss`).
    """

    def __init__(self, lambda_fil: float = 0.01, spatial_dims=None, temporal_dim=None):
        super().__init__()
        self.lambda_fil = lambda_fil
        self.nll = HeteroscedasticGaussianNLL()
        self.fil = FourierImputationLoss(spatial_dims=spatial_dims, temporal_dim=temporal_dim)

    def forward(self, input, target, mask=None, coords=None, pred_var=None, **kwargs):
        """
        Parameters
        ----------
        input : Tensor, shape [B, ..., F]
            Predicted means μ.
        target : Tensor, shape [B, ..., F]
            Ground truth with NaNs at missing locations.
        mask : Tensor or None
            Boolean - True at observed locations.
        pred_var : Tensor, shape [B, ..., F]
            Predicted log-variance log(σ²); same convention as
            :class:`HeteroscedasticGaussianNLL`.
        coords : ignored
            Accepted for API compatibility.
        """
        if pred_var is None:
            raise ValueError(
                "HeteroscedasticNLLWithFourierSparsity requires pred_var (log-variance). "
                "Make sure your model outputs a variance head."
            )

        # --- NLL on observed/masked points only --------------------------------
        nll_loss = self.nll(
            input=input,
            target=target,
            mask=mask,
            pred_var=pred_var,
        )
        if nll_loss is None:
            # No valid points - skip
            return None

        # --- FIL on the full composite field -----------------------------------
        fil_loss = self.fil(
            input=input,
            target=target,
            mask=mask,
        )

        total = nll_loss + self.lambda_fil * fil_loss

        return total

    def loss_components(self, input, target, mask=None, pred_var=None, **kwargs):
        """
        Returns individual loss components for logging/debugging.

        Returns:
            dict with keys 'nll', 'fil', 'total'
        """
        nll_loss = self.nll(input=input, target=target, mask=mask, pred_var=pred_var)
        fil_loss = self.fil(input=input, target=target, mask=mask)
        total = nll_loss + self.lambda_fil * fil_loss
        return {
            "nll": nll_loss.item(),
            "fil": fil_loss.item(),
            "total": total.item(),
        }
