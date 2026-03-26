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
    attention_type: str  # "mha" | "transformer_encoder_layer" | "encoder_decoder" | "space_time_attention" # @todo rename to transformer_encoder and autoencoder


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
            feature_mixer_input="feat",

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha"
        )},

    "exp4": {
        "description": "Raw KNN, coordinate encoder (model), MHA",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="raw",
            graph_metric="isotropic",

            encoder_scope="model",
            encoder_input="coords",
            encoder_output_dim=3,
            encoder_hidden_dim=64,

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat",  # Unused

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha"
        )},
    "exp5": {
        "description": "Raw KNN, coordinate encoder (graph), MHA",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="encoded",
            graph_metric="isotropic",

            encoder_scope="graph",
            encoder_input="coords",
            encoder_output_dim=3,
            encoder_hidden_dim=64,

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat",  # Unused

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha"
        )},
    "exp6": {
        "description": "Raw KNN, coordinate encoder (both), MHA",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="encoded",
            graph_metric="isotropic",

            encoder_scope="both",
            encoder_input="coords",
            encoder_output_dim=3,
            encoder_hidden_dim=64,

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat",

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha"
        )},
    "exp7": {
        "description": "Raw KNN, coordinate encoder (both, coords+feat), MHA",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="encoded",
            graph_metric="isotropic",

            encoder_scope="both",
            encoder_input="coords_feat",
            encoder_output_dim=3,
            encoder_hidden_dim=64,

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat",

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha"
        )},
    "exp8": {
        "description": "Raw KNN, coordinate encoder (both, coords+feat+mask), MHA",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="encoded",
            graph_metric="isotropic",

            encoder_scope="both",
            encoder_input="coords_feat_mask",
            encoder_output_dim=3,
            encoder_hidden_dim=64,

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat",

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha"
        )},
    "exp9": {
        "description": "Raw anisotropic KNN, MHA",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="raw",
            graph_metric="anisotropic",

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
    "exp10": {
        "description": "Raw KNN, transformer encoder",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="raw",
            graph_metric="isotropic",

            encoder_scope="none",
            encoder_input="coords",  # Unused
            encoder_output_dim=3,
            encoder_hidden_dim=64,

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat_mask",  # Unused

            use_rel_pos=False,
            use_masks=False,
            attention_type="transformer_encoder_layer"
        )},
    "exp11": {
        "description": "Raw KNN, encoder-decoder",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="raw",
            graph_metric="isotropic",

            encoder_scope="none",
            encoder_input="coords",  # Unused
            encoder_output_dim=3,
            encoder_hidden_dim=64,

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat_mask",  # Unused

            use_rel_pos=False,
            use_masks=False,
            attention_type="encoder_decoder"
        )},
    "exp12": {
        "description": "Dynamic, encoded KNN (coords), coordinate encoder (both), MHA",
        "config": ModelConfig(
            graph_mode="dynamic",
            graph_space="encoded",
            graph_metric="isotropic",

            encoder_scope="both",
            encoder_input="coords",
            encoder_output_dim=3,
            encoder_hidden_dim=64,

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat",

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha"
        )},
    "exp13": {
        "description": "Dynamic, encoded KNN (coord+feat), coordinate encoder (both), MHA",
        "config": ModelConfig(
            graph_mode="dynamic",
            graph_space="encoded",
            graph_metric="isotropic",

            encoder_scope="both",
            encoder_input="coords_feat",
            encoder_output_dim=3,
            encoder_hidden_dim=64,

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat",

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha"
        )},
    "exp14": {
        "description": "Dynamic, encoded KNN (coord+feat+mask), coordinate encoder (both), MHA",
        "config": ModelConfig(
            graph_mode="dynamic",
            graph_space="encoded",
            graph_metric="isotropic",

            encoder_scope="both",
            encoder_input="coords_feat_mask",
            encoder_output_dim=3,
            encoder_hidden_dim=64,

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat",

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha"
        )},
    "exp15": {
        "description": "Raw KNN, feature mixer (feat+mask), MHA",
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
            feature_mixer_input="feat_mask",

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha"
        )},
    "exp16": {
        "description": "Baseline, raw KNN, MHA, mean filling",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="raw",
            graph_metric="isotropic",

            encoder_scope="none",
            encoder_input="coords",  # Unused
            encoder_output_dim=3,  # Unused
            encoder_hidden_dim=64,  # Unused

            fill_strategy="mean",
            feature_mixer=False,
            feature_mixer_input="feat",  # Unused

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha"
        )},
    "exp17": {
        "description": "Baseline, raw KNN, space_time_attention",
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
            attention_type="space_time_attention"
        )},
    # "exp10": {
    #     "description": "Raw anisotropic KNN, MHA",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="encoded",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="graph",  # @todo coord encoder setting what was previously good?
    #         encoder_input="coords",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="zero",
    #         feature_mixer=False,
    #         feature_mixer_input="feat_mask",
    #
    #         use_rel_pos=False,
    #         use_masks=False,
    #         attention_type="mha"
    #     )},
    # "exp6": {
    #     "description": "Full model",
    #     "config": ModelConfig(
    #         graph_mode="dynamic",
    #         graph_space="encoded",
    #         graph_metric="isotropic",
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
    #         use_masks=False,
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
