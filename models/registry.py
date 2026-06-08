"""Model registry: maps model-name strings to classes or factory callables."""

from missingpy import MissForest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from hyperimpute.plugins.imputers import Imputers

from models.mastnet import MaSTNeT
from models.mlp import MLP
from remasker.remasker_impute import ReMasker


def get_model_class(model_name: str):
    """Return the model class or factory callable for *model_name*.

    Parameters
    ----------
    model_name : str
        One of the registered model keys (e.g. ``'mastnet'``, ``'knn'``).

    Returns
    -------
    type or callable
        The class itself for sklearn-style models, or a keyword-argument
        factory lambda for HyperImpute plugins.

    Raises
    ------
    ValueError
        If *model_name* is not in the registry.
    """
    imputers = Imputers()

    name_class_map = {
        "mean": SimpleImputer,
        "knn": KNNImputer,
        "missforest": MissForest,
        "mice": IterativeImputer,
        "mastnet": MaSTNeT,
        "remasker": ReMasker,
        "mlp": MLP,

        "hyperimpute": lambda **kwargs: imputers.get("hyperimpute", **kwargs),
        "gain_hyperimpute": lambda **kwargs: imputers.get("gain", **kwargs),
        "miracle_hyperimpute": lambda **kwargs: imputers.get("miracle", **kwargs),
        "miwae_hyperimpute": lambda **kwargs: imputers.get("miwae", **kwargs),
        "sklearn_ice_hyperimpute": lambda **kwargs: imputers.get("sklearn_ice", **kwargs),
        "nop_hyperimpute": lambda **kwargs: imputers.get("nop", **kwargs),
        "em_hyperimpute": lambda **kwargs: imputers.get("EM", **kwargs),
        "mice_hyperimpute": lambda **kwargs: imputers.get("mice", **kwargs),
        "softimpute_hyperimpute": lambda **kwargs: imputers.get("softimpute", **kwargs),
        "ice_hyperimpute": lambda **kwargs: imputers.get("ice", **kwargs),
        "missforest_hyperimpute": lambda **kwargs: imputers.get("missforest", **kwargs),
        "sklearn_missforest_hyperimpute": lambda **kwargs: imputers.get("sklearn_missforest", **kwargs),
        "sinkhorn_hyperimpute": lambda **kwargs: imputers.get("sinkhorn", **kwargs),
    }

    if model_name not in name_class_map:
        raise ValueError(f"Unknown model name: {model_name}")

    return name_class_map[model_name]
