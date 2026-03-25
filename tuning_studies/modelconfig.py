from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Graph
    graph_mode: str  # "static" | "dynamic"
    graph_space: str  # "raw" | "encoded"
    graph_metric: str  # "isotropic" | "anisotropic"

    # Encoder
    encoder_scope: str  # "none" | "graph" | "model" | "both"
    encoder_input: str  # "coords" | "coords_feat" | "coords_feat_mask"
    encoder_output_dim: int
    encoder_hidden_dim: int

    # Feature handling
    fill_strategy: str  # "zero" | "mean"
    feature_mixer: bool
    feature_mixer_input: str  # "feat" | "feat_mask"

    # Attention
    use_rel_pos: bool
    use_masks: bool
    attention_type: str  # "mha" | "transformer_encoder_layer" | "encoder_decoder"


ablation_study = {
    "exp0": {
        "description": "Baseline, raw KNN, MHA",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="raw",
            graph_metric="isotropic",

            encoder_scope="none",
            encoder_input="coords",  # Unused
            encoder_output_dim=3,  # Unused
            encoder_hidden_dim=64,  # Unused

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat",  # Unused

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha"
        )},
    "exp1": {
        "description": "Raw KNN, MHA with masks",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="raw",
            graph_metric="isotropic",

            encoder_scope="none",
            encoder_input="coords",  # Unused
            encoder_output_dim=3,  # Unused
            encoder_hidden_dim=64,  # Unused

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat",  # Unused

            use_rel_pos=False,
            use_masks=True,
            attention_type="mha"
        )},
    "exp2": {
        "description": "Raw KNN, MHA with rel_pos",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="raw",
            graph_metric="isotropic",

            encoder_scope="none",
            encoder_input="coords",  # Unused
            encoder_output_dim=3,  # Unused
            encoder_hidden_dim=64,  # Unused

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat",  # Unused

            use_rel_pos=True,
            use_masks=False,
            attention_type="mha"
        )},

    "exp3": {
        "description": "Raw KNN, feature mixer, MHA",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="raw",
            graph_metric="isotropic",

            encoder_scope="none",
            encoder_input="coords",  # Unused
            encoder_output_dim=3,  # Unused
            encoder_hidden_dim=64,  # Unused

            fill_strategy="zero",
            feature_mixer=True,
            feature_mixer_input="feat",  # Unused

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha"
        )},

    "exp4": {
        "description": "Raw KNN, coordinate encoder (model) feature mixer, MHA",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="raw",
            graph_metric="isotropic",

            encoder_scope="model",
            encoder_input="coords",  # Unused
            encoder_output_dim=3,  # Unused
            encoder_hidden_dim=64,  # Unused

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat",  # Unused

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha"
        )},
    "exp5": {
        "description": "Raw KNN, coordinate encoder (graph) feature mixer, MHA",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="encoded",
            graph_metric="isotropic",

            encoder_scope="graph",
            encoder_input="coords",  # Unused
            encoder_output_dim=3,  # Unused
            encoder_hidden_dim=64,  # Unused

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat",  # Unused

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha"
        )},
    "exp6": {
        "description": "Raw anisotropic KNN, MHA",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="raw",
            graph_metric="anisotropic",  # Unused

            encoder_scope="none",
            encoder_input="coords_feat_mask",
            encoder_output_dim=3,
            encoder_hidden_dim=64,

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat_mask",

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha"
        )},
    # "exp6": {
    #     "description": "Full model",
    #     "config": ModelConfig(
    #         graph_mode="dynamic",
    #         graph_space="encoded",
    #         graph_metric="isotropic",  # Unused
    #
    #         encoder_scope="both",
    #         encoder_input="coords_feat_mask",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="mean",
    #         feature_mixer=True,
    #         feature_mixer_input="feat_mask",
    #
    #         use_rel_pos=True,
    #         use_masks=True,
    #         attention_type="mha"
    #     )},
}

# # Required combinations
# graph_space = "raw",
# graph_metric = "isotropic", --> KNN only on coords ok
#
# graph_space = "raw",
# graph_metric = "anisotropic", --> anisotropic KNN
#
# graph_space = "encoded",
# graph_metric = "isotropic", --> Test encoder
