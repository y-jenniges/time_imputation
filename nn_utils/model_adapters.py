from abc import ABC, abstractmethod
import torch

from nn_utils.dataset import random_feature_mask, transect_feature_mask, spherical_feature_mask, \
    random_per_feature_mask, random_token_mask


class ModelAdapter(ABC):
    @abstractmethod
    def batch_size(self, batch):
        pass

    @abstractmethod
    def prepare_batch(self, batch, device):
        pass

    @abstractmethod
    def make_masks(self, batch, mask_ratio, mode="train", device:torch.device = torch.device("cpu"), cfg=None):
        pass

    @abstractmethod
    def forward(self, model, batch, masks):
        pass

    @abstractmethod
    def loss_inputs(self, batch, outputs, masks, **kwargs):
        pass

    @abstractmethod
    def outputs_to_cpu(self, batch, outputs, to_numpy: bool = True):
        pass

    @abstractmethod
    def get_query_mask(self, batch):
        pass


class NeighbourAdapter(ModelAdapter):
    def batch_size(self, batch):
        return batch["query_features"].shape[0]

    def prepare_batch(self, batch, device):
        return self.to_device(batch, device)

    def to_device(self, x, device):
        if isinstance(x, dict):
            return {k: self.to_device(v, device) for k, v in x.items()}
        return x.to(device)

    def make_masks(self, batch, mask_ratio=0.0, mode="train", device=torch.device("cpu"), cfg=None):
        masking_strategies = ["random"] if cfg.masking_strategies is None else cfg.masking_strategies
        q_mask = batch["query_mask"]
        batch_size = q_mask.shape[0]

        scopes = [scope for scope in batch.keys() if scope not in ["query_features", "query_mask", "query_coords"]]
        masks = {}

        # Query masking
        if mode in ["train", "eval"] and mask_ratio > 0:
            masks_list = []
            if "random" in masking_strategies:
                q_rand_mask = torch.zeros(q_mask.shape, device=device)
                for i in range(batch_size):
                    q_rand_mask, _ = random_feature_mask(
                        batch_size=q_mask.shape[0],
                        feature_dim=q_mask.shape[1],
                        mask_ratio=mask_ratio,
                        device=device,
                        mask_query=True,
                        mask_neighbours=False
                    )
                    if q_rand_mask.any():
                        break
                masks_list.append(q_rand_mask)

            if "per_feature" in masking_strategies:
                q_per_feature_mask = torch.zeros(q_mask.shape, device=device)
                for i in range(batch_size):
                    q_per_feature_mask = random_per_feature_mask(
                        batch_size=q_mask.shape[0],
                        feature_dim=q_mask.shape[1],
                        mask_ratio=cfg.mask_ratio,
                        device=device)
                    if q_per_feature_mask.any():
                        break
                masks_list.append(q_per_feature_mask)

            if "transect" in masking_strategies:
                q_transect_mask = torch.zeros(q_mask.shape, device=device)
                for i in range(batch_size):
                    q_transect_mask = transect_feature_mask(batch=batch,feature_dim=q_mask.shape[1],
                                                            width=cfg.transect_mask_width,
                                                            p=cfg.transect_mask_p,
                                                            orientation=cfg.transect_mask_orientation,
                                                            device=device)
                    if q_transect_mask.any():
                        break
                masks_list.append(q_transect_mask)

            if "sphere" in masking_strategies:
                q_sphere_mask = torch.zeros(q_mask.shape, device=device)
                for i in range(batch_size):
                    q_sphere_mask = spherical_feature_mask(batch=batch, feature_dim=q_mask.shape[1],
                                                          size=cfg.sphere_mask_radius, p=cfg.sphere_mask_p, device=device)
                    if q_sphere_mask.any():
                        break
                masks_list.append(q_sphere_mask)

            # Combine masks
            q_random_mask = masks_list[0]
            if len(masks_list) > 1:
                for mask in masks_list[1:]:
                    q_random_mask = q_random_mask | mask
            elif len(masks_list) == 0:
                raise ValueError("Please specify at least one masking strategy.")

            masks["q_input_mask"] = q_mask & ~q_random_mask  # Observed inputs after random masking
            masks["q_loss_mask"] = q_mask & q_random_mask  # Positions to reconstruct: Originally observed but randomly hidden
        elif mode == "reconstruct":
            # Reconstruct all missing features, feed all observed
            masks["q_input_mask"] = q_mask
            masks["q_loss_mask"] = torch.zeros_like(q_mask, dtype=torch.bool)
        else:
            masks["q_input_mask"] = q_mask
            masks["q_loss_mask"] = q_mask

        # Neighbour masking (per scope)
        masks["neighbour_input"] = {}
        masks["neighbour_loss"] = {}
        for scope in scopes:
            n_mask = batch[scope]["mask"]

            if mode in ["train", "eval"] and mask_ratio > 0:
                _, n_random_mask = random_feature_mask(
                    batch_size=q_mask.shape[0],
                    feature_dim=q_mask.shape[1],
                    mask_ratio=mask_ratio,
                    n_neighbours=n_mask.shape[1],
                    device=device,
                    mask_query=False,
                    mask_neighbours=True
                )
                masks["neighbour_input"][scope] = n_mask & ~n_random_mask
                masks["neighbour_loss"][scope] = n_mask & n_random_mask

            elif mode == "reconstruct":
                masks["neighbour_input"][scope] = n_mask
                masks["neighbour_loss"][scope] = torch.zeros_like(n_mask, dtype=torch.bool)
            else:
                masks["neighbour_input"][scope] = n_mask
                masks["neighbour_loss"][scope] = n_mask

        # Compute missingness ratio for query token
        total_valid = q_mask.sum()  # Total valid entries
        total_masked = masks["q_loss_mask"].sum()  # Actually masked entries (only where data existed)

        # Avoid division by zero
        if total_valid > 0:
            miss_ratio = total_masked.float() / total_valid.float()
        else:
            miss_ratio = torch.tensor(0.0, device=q_mask.device)
        return masks, miss_ratio

    def forward(self, model, batch, masks):
        batch["query_mask"] = masks["q_input_mask"]

        for scope, m in masks["neighbour_input"].items():
            batch[scope]["mask"] = m

        return model(batch)

    def loss_inputs(self, batch, outputs, masks, **kwargs):
        pred_mean, pred_var = outputs

        return dict(
            input=pred_mean,
            target=batch["query_features"],
            mask=masks["q_loss_mask"],
            pred_var=pred_var,
            query_coords=batch["query_coords"],
            anisotropic_weights=kwargs.get("anisotropic_weights", None),

            # Structured neighbour info
            neighbours={
                scope: {
                    "features": batch[scope]["features"],
                    "coords": batch[scope]["coords"],
                    "mask": masks["neighbour_loss"][scope]
                }
                for scope in batch.keys() if scope not in ["query_features", "query_mask", "query_coords"]
            }
        )

    def outputs_to_cpu(self, batch, outputs, to_numpy: bool = True):
        pred_mean, _ = outputs

        q_feat = batch["query_features"].detach().cpu()
        pred_mean = pred_mean.detach().cpu()

        if to_numpy:
            return q_feat.numpy(), pred_mean.numpy()
        else:
            return q_feat, pred_mean

    def get_query_mask(self, batch):
        return batch["query_mask"]


class TimeSequenceAdapter(ModelAdapter):
    def batch_size(self, batch):
        return batch["query_features"].shape[0]

    def prepare_batch(self, batch, device):
        return self.to_device(batch, device)

    def to_device(self, x, device):
        if isinstance(x, dict):
            return {k: self.to_device(v, device) for k, v in x.items()}
        return x.to(device)

    def make_masks(self, batch, mask_ratio=0.0, mode="train", device=torch.device("cpu"), cfg=None):
        q_mask = batch["query_mask"]

        scopes = [scope for scope in batch.keys() if scope not in ["query_features", "query_mask", "query_coords"]]
        masks = {}

        # Query masking
        if mode in ["train", "eval"] and mask_ratio > 0:
            batch_size = batch["query_mask"].shape[0]
            q_random_mask = torch.zeros(q_mask.shape, device=device)
            for i in range(batch_size):
                q_random_mask, _ = random_feature_mask(
                    batch_size=q_mask.shape[0],
                    feature_dim=q_mask.shape[1],
                    mask_ratio=mask_ratio,
                    device=device,
                    mask_query=True,
                    mask_neighbours=False
                )
                if q_random_mask.any():
                    break

            masks["q_input_mask"] = q_mask & ~q_random_mask  # Observed inputs after random masking
            masks["q_loss_mask"] = q_mask & q_random_mask  # Positions to reconstruct: Originally observed but randomly hidden
        elif mode == "reconstruct":
            # Reconstruct all missing features, feed all observed
            masks["q_input_mask"] = q_mask
            masks["q_loss_mask"] = torch.zeros_like(q_mask, dtype=torch.bool)
        else:
            masks["q_input_mask"] = q_mask
            masks["q_loss_mask"] = q_mask

        # Neighbour masking (per scope)
        masks["sequence_input"] = {}
        masks["sequence_loss"] = {}
        for scope in scopes:
            n_mask = batch[scope]["mask"]

            if mode in ["train", "eval"] and mask_ratio > 0:
                _, n_random_mask = random_feature_mask(
                    batch_size=q_mask.shape[0],
                    feature_dim=q_mask.shape[1],
                    mask_ratio=mask_ratio,
                    n_neighbours=n_mask.shape[1],
                    device=device,
                    mask_query=False,
                    mask_neighbours=True
                )
                masks["sequence_input"][scope] = n_mask & ~n_random_mask
                masks["sequence_loss"][scope] = n_mask & n_random_mask

            elif mode == "reconstruct":
                masks["sequence_input"][scope] = n_mask
                masks["sequence_loss"][scope] = torch.zeros_like(n_mask, dtype=torch.bool)
            else:
                masks["sequence_input"][scope] = n_mask
                masks["sequence_loss"][scope] = n_mask

        # Compute missingness ratio for query token
        total_valid = q_mask.sum()  # Total valid entries
        total_masked = masks["q_loss_mask"].sum()  # Actually masked entries (only where data existed)

        # Avoid division by zero
        if total_valid > 0:
            miss_ratio = total_masked.float() / total_valid.float()
        else:
            miss_ratio = torch.tensor(0.0, device=q_mask.device)

        return masks, miss_ratio

    def forward(self, model, batch, masks):
        batch["query_mask"] = masks["q_input_mask"]

        for scope, seq_mask in masks["sequence_input"].items():
            # Replace center position with query mask
            center_pos = seq_mask.shape[1] // 2
            sm = seq_mask.clone()  # Break potential shared memory
            sm[:, center_pos, :] = masks["q_input_mask"]

            batch[scope]["mask"] = sm

        return model(batch)

    def loss_inputs(self, batch, outputs, masks, **kwargs):
        pred_mean, pred_var = outputs

        return dict(
            input=pred_mean,
            target=batch["query_features"],
            mask=masks["q_loss_mask"],
            pred_var=pred_var,
            query_coords=batch["query_coords"],
            anisotropic_weights=kwargs.get("anisotropic_weights", None),

            # Structured sequence info
            sequence={
                scope: {
                    "features": batch[scope]["features"],
                    "coords": batch[scope]["coords"],
                    "mask": masks["sequence_loss"][scope]
                }
                for scope in batch.keys() if scope not in ["query_features", "query_mask", "query_coords"]
            }
        )

    def outputs_to_cpu(self, batch, outputs, to_numpy: bool = True):
        pred_mean, _ = outputs

        q_feat = batch["query_features"].detach().cpu()
        pred_mean = pred_mean.detach().cpu()

        if to_numpy:
            return q_feat.numpy(), pred_mean.numpy()
        else:
            return q_feat, pred_mean

    def get_query_mask(self, batch):
        return batch["query_mask"]


class FeatureAdapter(ModelAdapter):
    def batch_size(self, batch):
        return batch["features"].shape[0]

    def prepare_batch(self, batch, device):
        return {k: v.to(device) for k, v in batch.items()}

    def make_masks(self, batch, mask_ratio, mode="train", device=torch.device("cpu"), masking_strategies=None, cfg=None):
        feat = batch["features"]
        mask = batch["mask"]
        batch_size, n_features = feat.shape

        if mode in ["train", "eval"] and mask_ratio > 0:
            # random_mask: True means "mask/hide this token"
            random_mask = random_token_mask(batch_size=batch_size, feature_dim=n_features, mask_ratio=mask_ratio, device=device)

            # Observed inputs after random masking
            input_mask = mask & ~random_mask

            # Positions to reconstruct: originally observed but randomly hidden
            loss_mask = mask & random_mask

        elif mode == "reconstruct":
            # Reconstruct all missing features
            input_mask = mask
            loss_mask = torch.zeros_like(mask, dtype=torch.bool)

        else:
            input_mask = mask
            loss_mask = mask

        # Compute missingness ratio for query token
        total_valid = input_mask.sum()  # Total valid entries
        total_masked = loss_mask.sum()  # Actually masked entries (only where data existed)

        # Avoid division by zero
        if total_valid > 0:
            miss_ratio = total_masked.float() / total_valid.float()
        else:
            miss_ratio = torch.tensor(0.0, device=loss_mask.device)

        return dict(input_mask=input_mask, loss_mask=loss_mask), miss_ratio

    # def forward(self, model, batch, masks):
    #     return model(batch)

    def forward(self, model, batch, masks):
        batch["mask"] = masks["input_mask"]
        return model(batch)

    def loss_inputs(self, batch, outputs, masks, **kwargs):
        if len(outputs) == 2:
            pred_mean, pred_var = outputs
        else:
            pred_mean = outputs
            pred_var = None
        return dict(input=pred_mean, target=batch["features"], mask=masks["loss_mask"], pred_var=pred_var)

    def outputs_to_cpu(self, batch, outputs, to_numpy: bool = True):
        if len(outputs) == 2:
            pred_mean, _ = outputs
        else:
            pred_mean = outputs

        q_feat = batch["features"].detach().cpu()
        pred_mean = pred_mean.detach().cpu()

        if to_numpy:
            return q_feat.numpy(), pred_mean.numpy()
        else:
            return q_feat, pred_mean

    def get_query_mask(self, batch):
        return batch["mask"]


class PointwiseAdapter(ModelAdapter):
    def batch_size(self, batch):
        return batch["features"].shape[0]

    def prepare_batch(self, batch, device):
        return {k: v.to(device) for k, v in batch.items()}

    def make_masks(self, batch, mask_ratio, mode="train", device=torch.device("cpu"), masking_strategies=None, cfg=None):
        feat = batch["features"]
        mask = batch["mask"]
        batch_size, n_features = feat.shape

        if mode in ["train", "eval"] and mask_ratio > 0:
            # random_feature_mask: True means "mask/hide this feature"
            random_mask, _ = (
                random_feature_mask(batch_size=batch_size, feature_dim=n_features, mask_ratio=mask_ratio,
                                    n_neighbours=0, device=device, mask_query=True, mask_neighbours=False))

            # Observed inputs after random masking
            input_mask = mask & ~random_mask

            # Positions to reconstruct: originally observed but randomly hidden
            loss_mask = mask & random_mask

        elif mode == "reconstruct":
            # Reconstruct all missing features
            input_mask = mask
            loss_mask = torch.zeros_like(mask, dtype=torch.bool)

        else:
            input_mask = mask
            loss_mask = mask

        # Compute missingness ratio for query token
        total_valid = input_mask.sum()  # Total valid entries
        total_masked = loss_mask.sum()  # Actually masked entries (only where data existed)

        # Avoid division by zero
        if total_valid > 0:
            miss_ratio = total_masked.float() / total_valid.float()
        else:
            miss_ratio = torch.tensor(0.0, device=loss_mask.device)

        return dict(input_mask=input_mask, loss_mask=loss_mask), miss_ratio

    def forward(self, model, batch, masks):
        feat = batch["features"]
        mask = masks["input_mask"]
        coords = batch["coords"]

        feat_filled = torch.where(mask, feat, torch.zeros_like(feat))
        x = torch.cat([coords, feat_filled, mask.float()], dim=-1)
        return model(x)

    def loss_inputs(self, batch, outputs, masks, **kwargs):
        if len(outputs) == 2:
            pred_mean, pred_var = outputs
        else:
            pred_mean = outputs
            pred_var = None
        return dict(input=pred_mean, target=batch["features"], mask=masks["loss_mask"], pred_var=pred_var)

    def outputs_to_cpu(self, batch, outputs, to_numpy: bool = True):
        if len(outputs) == 2:
            pred_mean, _ = outputs
        else:
            pred_mean = outputs

        q_feat = batch["features"].detach().cpu()
        pred_mean = pred_mean.detach().cpu()

        if to_numpy:
            return q_feat.numpy(), pred_mean.numpy()
        else:
            return q_feat, pred_mean

    def get_query_mask(self, batch):
        return batch["mask"]
