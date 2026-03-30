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
    attention_type: str  # "mha" | "transformer_encoder" | "autoencoder" | "space_time_attention"

    n_neighbours: int = 30
    graph_update_frequency: int = 5
    graph_warmup: int = 0
    graph_freeze_epoch: int = 20
    learn_anisotropic_weights: bool = False
    positional_encoding: bool = False
    n_time_layers: int = 3
    loss_name: str = "hetero"


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
            attention_type="transformer_encoder"
        )},
    "exp11": {
        "description": "Raw KNN, autoencoder",
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
            attention_type="autoencoder"
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
    "exp18": {
        "description": "Baseline, raw KNN, space_time_depth_attention",
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
            attention_type="space_time_depth_attention"
        )},
    "exp19": {
        "description": "Baseline, raw KNN, MHA, k=5",
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
            attention_type="mha",

            n_neighbours = 5
        )},
    "exp20": {
        "description": "Baseline, raw KNN, MHA, k=10",
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
            attention_type="mha",

            n_neighbours=10
        )},
    "exp21": {
        "description": "Baseline, raw KNN, MHA, k=20",
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
            attention_type="mha",

            n_neighbours=20
        )},

    "exp22": {
        "description": "Baseline, raw KNN, MHA, k=50",
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
            attention_type="mha",

            n_neighbours=50
        )},

    "exp23": {
        "description": "Baseline, raw KNN, MHA, k=100",
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
            attention_type="mha",

            n_neighbours=100
        )},

    "exp24": {
        "description": "Baseline, raw KNN, MHA, positional encoding",
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
            attention_type="mha",

            positional_encoding=True
        )},
    "exp25": {
        "description": "Raw KNN, feature mixer (feat+mask), MHA with masks",
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
            use_masks=True,
            attention_type="mha"
        )},
    "exp26": {
        "description": "Raw KNN, feature mixer (feat+mask), MHA, rel_pos",
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

            use_rel_pos=True,
            use_masks=False,
            attention_type="mha"
        )},
    "exp27": {
        "description": "Raw KNN, feature mixer (feat+mask), autoencoder",
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
            attention_type="autoencoder"
        )},
    "exp28": {
        "description": "Raw KNN, feature mixer (feat+mask), autoencoder with masks",
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
            use_masks=True,
            attention_type="autoencoder"
        )},
    "exp29": {
        "description": "Raw KNN, feature mixer (feat+mask), autoencoder, rel_pos",
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

            use_rel_pos=True,
            use_masks=False,
            attention_type="autoencoder"
        )},
    "exp30": {
        "description": "Raw KNN, feature mixer (feat), MHA with masks",
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
            use_masks=True,
            attention_type="mha"
        )},
    "exp31": {
        "description": "Raw KNN, feature mixer (feat+mask), space_time_attention",
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
            attention_type="space_time_attention"
        )},
    "exp32": {
        "description": "Raw KNN, space_time_attention (#time_layers=1)",
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
            feature_mixer_input="feat_mask",

            use_rel_pos=False,
            use_masks=False,
            attention_type="space_time_attention",

            n_time_layers = 1
        )},
    "exp33": {
        "description": "Raw KNN, MHA, weighted time???????????????? todo",
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
            feature_mixer_input="feat_mask",

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha",
        )},
    "exp34": {
        "description": "Dynamic KNN, MHA, update every 10 epochs (until epoch 50)",
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
            feature_mixer_input="feat_mask",

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha",

            graph_update_frequency=10,
            graph_freeze_epoch=50,
        )},
    "exp35": {
        "description": "Dynamic KNN, MHA, update every epoch",
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
            feature_mixer_input="feat_mask",

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha",

            graph_update_frequency=1
        )},
    "exp36": {
        "description": "Raw KNN, MHA, 20 epochs warmup before graph learning",
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
            feature_mixer_input="feat_mask",

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha",

            graph_warmup=20,
            graph_freeze_epoch=200,  # No freezing
        )},
    "exp37": {
        "description": "Raw KNN, autoencoder, feature mixer (feat)",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="raw",
            graph_metric="isotropic",

            encoder_scope="none",
            encoder_input="coords",
            encoder_output_dim=3,
            encoder_hidden_dim=64,

            fill_strategy="zero",
            feature_mixer=True,
            feature_mixer_input="feat",

            use_rel_pos=False,
            use_masks=False,
            attention_type="autoencoder"
        )},
    "exp38": {
        "description": "Raw KNN, autoencoder with masks, feature mixer (feat)",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="raw",
            graph_metric="isotropic",

            encoder_scope="none",
            encoder_input="coords",
            encoder_output_dim=3,
            encoder_hidden_dim=64,

            fill_strategy="zero",
            feature_mixer=True,
            feature_mixer_input="feat",

            use_rel_pos=False,
            use_masks=True,
            attention_type="autoencoder",
        )},
    "exp39": {
        "description": "Raw KNN, space_time_attention (#time_layers=2)",
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
            feature_mixer_input="feat_mask",

            use_rel_pos=False,
            use_masks=False,
            attention_type="space_time_attention",

            n_time_layers=2
        )},
    "exp40": {
        "description": "Dynamic KNN, update every epoch (no freeze)",
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
            feature_mixer_input="feat_mask",

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha",

            graph_update_frequency=1,
            graph_freeze_epoch=200
        )},
    "exp41": {
        "description": "Raw KNN, MHA, 20 epochs warmup before graph learning (update every epoch, no freeze)",
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
            feature_mixer_input="feat_mask",

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha",

            graph_warmup=20,
            graph_update_frequency=1,
            graph_freeze_epoch=200,  # No freezing
        )},
    "exp42": {
        "description": "Raw KNN, MHA, graph_dim=16 (not 3)",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="raw",
            graph_metric="isotropic",

            encoder_scope="none",
            encoder_input="coords",
            encoder_output_dim=16,
            encoder_hidden_dim=64,

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat_mask",

            use_rel_pos=False,
            use_masks=False,
            attention_type="mha",
        )},
    "exp43": {
        "description": "Raw KNN (learnable dim weights), MHA",
        "config": ModelConfig(
            graph_mode="dynamic",
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
            attention_type="mha",

            learn_anisotropic_weights=True,
            graph_update_frequency=1,
            graph_freeze_epoch=200,
            loss_name="hetero_smooth",
        )},
    "exp44": {
        "description": "Raw KNN, MHA, hetero smooth loss",
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
            attention_type="mha",

            graph_update_frequency=1,
            graph_freeze_epoch=200,
            loss_name="hetero_smooth",
        )},

    # "exp19": {
    #     "description": "Baseline, raw KNN, space_time_attention (only 1 time layer, not 3)",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="raw",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="none",
    #         encoder_input="coords",  # Unused
    #         encoder_output_dim=3,  # Unused
    #         encoder_hidden_dim=64,  # Unused
    #
    #         fill_strategy="zero",
    #         feature_mixer=False,
    #         feature_mixer_input="feat",  # Unused
    #
    #         use_rel_pos=False,
    #         use_masks=False,
    #         attention_type="space_time_attention"
    #     )},


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
