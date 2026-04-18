from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Graph
    graph_mode: str = "static"  # "static" | "dynamic" | "random" | "time_sequence"  (with time_sequence, time sequence input will be used, no neighbourhoods graph)
    graph_space: str = "raw"  # "raw" | "encoded"
    graph_metric: str = "isotropic"  # "isotropic" | "anisotropic"

    n_neighbours: int = 30
    graph_update_frequency: int = 5
    graph_warmup: int = 0
    graph_freeze_epoch: int = 20
    learn_anisotropic_weights: bool = False

    # Coordinate encoder
    encoder_scope: str ="none"  # "none" | "graph" | "model" | "both"
    encoder_input: str  = "coords"  # "coords" | "coords_feat" | "coords_feat_mask"
    encoder_output_dim: int = 3
    encoder_hidden_dim: int = 64

    # Feature handling
    fill_strategy: str = "zero"  # "zero" | "mean"
    feature_mixer: bool = "False"
    feature_mixer_input: str = "feat"  # "feat" | "feat_mask"

    # Attention
    use_rel_pos: bool = False
    use_masks: bool = False
    attention_type: str = "mha"  # "mha" | "transformer_encoder" | "autoencoder" | "space_time_attention" | "space_time_depth_attention" | "weighted_space_time_depth_attention"
    n_time_layers: int = 3
    global_context: bool = False

    # Masking
    masking_strategies: list[str] = None  # "random" | "transect" | "sphere" | "per_feature"
    sphere_mask_radius: float = 0.1
    sphere_mask_p: float = 0.3
    transect_mask_width: float = 0.05
    transect_mask_p: float = 0.3
    transect_mask_orientation: float = 0.0
    mask_ratio: float = 0.3

    # Positional encoding
    positional_encoding: bool = False
    positional_encoding_time_only: bool = False

    # Loss
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
        "description": "Raw KNN, MHA, graph_dim=16 (not 3) with encoder [both on coords]",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="raw",
            graph_metric="isotropic",

            encoder_scope="both",
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

            loss_name="hetero_smooth",
        )},
    "exp45": {
        "description": "Raw KNN (learnable dim weights using exp not softplus), MHA",
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
    "exp46": {
        "description": "Random neighbours",
        "config": ModelConfig(
            graph_mode="random",
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
        )},
    "exp47": {
        "description": "Time sequence, autoencoder",
        "config": ModelConfig(
            graph_mode="time_sequence",
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
            attention_type="autoencoder",

            positional_encoding_time_only=True,
        )},
    "exp48": {
        "description": "Raw KNN, MHA-Decoder",
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
            attention_type="mha_decoder"
        )},
    "exp49": {
        "description": "Raw KNN, autoencoder with masks, feature mixer (feat); hetero_smooth (learnable anisotropic weights)",
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

            loss_name="hetero_smooth",
            learn_anisotropic_weights=True,
        )},
    "exp50": {
        "description": "Raw KNN, autoencoder with masks, feature mixer (feat), global context",
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

            global_context=True,
        )},
    "exp51": {
        "description": "Raw KNN, autoencoder with masks, global context",
        "config": ModelConfig(
            graph_mode="static",
            graph_space="raw",
            graph_metric="isotropic",

            encoder_scope="none",
            encoder_input="coords",
            encoder_output_dim=3,
            encoder_hidden_dim=64,

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat",

            use_rel_pos=False,
            use_masks=True,
            attention_type="autoencoder",

            global_context=True,
        )},
    "exp52": {
        "description": "Raw KNN, autoencoder with masks, feature mixer (feat), gated global context",
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

            global_context=True,
        )},
    # "exp53": {
    #     "description": "Raw KNN, MHA, random masking + transects",
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
    #         attention_type="mha"
    #     )},
    # "exp54": {
    #     "description": "Raw KNN, autoencoder with masks, feature mixer (feat), random masking + transects",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="raw",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="none",
    #         encoder_input="coords",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="zero",
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #
    #         use_rel_pos=False,
    #         use_masks=True,
    #         attention_type="autoencoder",
    #
    #         masking_strategies=["random", "line"]
    #     )},
    # "exp55": {
    #     "description": "Raw KNN, feature mixer (feat+mask), space_time_attention, random masking + transects",
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
    #         feature_mixer=True,
    #         feature_mixer_input="feat_mask",
    #
    #         use_rel_pos=False,
    #         use_masks=False,
    #         attention_type="space_time_attention",
    #
    #         masking_strategies=["random", "line"],
    #     )},
    # "exp56": {
    #     "description": "Raw KNN, feature mixer (feat+mask), space_time_attention, random masking + transects + blocks",
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
    #         feature_mixer=True,
    #         feature_mixer_input="feat_mask",
    #
    #         use_rel_pos=False,
    #         use_masks=False,
    #         attention_type="space_time_attention",
    #
    #         masking_strategies=["random", "line", "block"],
    #     )},
    # "exp57": {
    #     "description": "Raw KNN, autoencoder with masks, feature mixer (feat), random masking + transects + blocks",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="raw",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="none",
    #         encoder_input="coords",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="zero",
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #
    #         use_rel_pos=False,
    #         use_masks=True,
    #         attention_type="autoencoder",
    #
    #         masking_strategies=["random", "line", "block"]
    #     )},
    #
    # "exp58": {
    #     "description": "Raw KNN, autoencoder with masks, feature mixer (feat), random masking + blocks",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="raw",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="none",
    #         encoder_input="coords",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="zero",
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #
    #         use_rel_pos=False,
    #         use_masks=True,
    #         attention_type="autoencoder",
    #
    #         masking_strategies=["random", "block"]
    #     )},
    # "exp59": {
    #     "description": "Raw KNN, autoencoder with masks, feature mixer (feat), transect masking",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="raw",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="none",
    #         encoder_input="coords",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="zero",
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #
    #         use_rel_pos=False,
    #         use_masks=True,
    #         attention_type="autoencoder",
    #
    #         masking_strategies=["line"],
    #         line_mask_p=1.0,
    #     )},
    # "exp60": {
    #     "description": "Raw KNN, autoencoder with masks, feature mixer (feat), block masking",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="raw",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="none",
    #         encoder_input="coords",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="zero",
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #
    #         use_rel_pos=False,
    #         use_masks=True,
    #         attention_type="autoencoder",
    #
    #         masking_strategies=["block"],
    #         sphere_mask_p=1.0,
    #     )},

    # "exp61": {
    #     "description": "Raw KNN, autoencoder with masks, feature mixer (feat), random+block masking p=1.0",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="raw",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="none",
    #         encoder_input="coords",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="zero",
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #
    #         use_rel_pos=False,
    #         use_masks=True,
    #         attention_type="autoencoder",
    #
    #         masking_strategies=["random", "block"],
    #         sphere_mask_p=1.0,
    #     )},
    #
    # "exp62": {
    #     "description": "Raw KNN, autoencoder with masks, feature mixer (feat), random+transect masking p=1.0",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="raw",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="none",
    #         encoder_input="coords",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="zero",
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #
    #         use_rel_pos=False,
    #         use_masks=True,
    #         attention_type="autoencoder",
    #
    #         masking_strategies=["random", "line"],
    #         transect_mask_p=1.0,
    #     )},
    #
    # "exp63": {
    #     "description": "Raw KNN, autoencoder with masks, feature mixer (feat), random+block masking p=1.0, s=0.2",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="raw",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="none",
    #         encoder_input="coords",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="zero",
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #
    #         use_rel_pos=False,
    #         use_masks=True,
    #         attention_type="autoencoder",
    #
    #         masking_strategies=["random", "block"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.2,
    #     )},
    # "exp64": {
    #     "description": "Raw KNN, autoencoder with masks, feature mixer (feat), random+block masking p=1.0, s=0.3",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="raw",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="none",
    #         encoder_input="coords",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="zero",
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #
    #         use_rel_pos=False,
    #         use_masks=True,
    #         attention_type="autoencoder",
    #
    #         masking_strategies=["random", "block"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.3,
    #     )},
    # "exp65": {
    #     "description": "Raw KNN, autoencoder with masks, feature mixer (feat), random+block masking p=1.0, s=0.4",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="raw",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="none",
    #         encoder_input="coords",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="zero",
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #
    #         use_rel_pos=False,
    #         use_masks=True,
    #         attention_type="autoencoder",
    #
    #         masking_strategies=["random", "block"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.4,
    #     )},
    # "exp66": {
    #     "description": "Raw KNN, autoencoder with masks, feature mixer (feat), random+block masking p=1.0, s=0.5",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="raw",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="none",
    #         encoder_input="coords",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="zero",
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #
    #         use_rel_pos=False,
    #         use_masks=True,
    #         attention_type="autoencoder",
    #
    #         masking_strategies=["random", "block"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.5,
    #     )},
    # "exp67": {
    #     "description": "Raw KNN, autoencoder with masks, feature mixer (feat), random+block masking p=1.0, s=0.6",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="raw",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="none",
    #         encoder_input="coords",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="zero",
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #
    #         use_rel_pos=False,
    #         use_masks=True,
    #         attention_type="autoencoder",
    #
    #         masking_strategies=["random", "block"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.6,
    #     )},
    # "exp68": {
    #     "description": "Raw KNN, autoencoder with masks, feature mixer (feat), random+block masking p=1.0, s=0.7",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="raw",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="none",
    #         encoder_input="coords",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="zero",
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #
    #         use_rel_pos=False,
    #         use_masks=True,
    #         attention_type="autoencoder",
    #
    #         masking_strategies=["random", "block"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.7,
    #     )},
    # "exp69": {
    #     "description": "Raw KNN, autoencoder with masks, feature mixer (feat), random+block masking p=1.0, s=0.8",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="raw",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="none",
    #         encoder_input="coords",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="zero",
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #
    #         use_rel_pos=False,
    #         use_masks=True,
    #         attention_type="autoencoder",
    #
    #         masking_strategies=["random", "block"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.8,
    #     )},
    # "exp70": {
    #     "description": "Raw KNN, autoencoder with masks, feature mixer (feat), random+block masking p=1.0, s=0.9",
    #     "config": ModelConfig(
    #         graph_mode="static",
    #         graph_space="raw",
    #         graph_metric="isotropic",
    #
    #         encoder_scope="none",
    #         encoder_input="coords",
    #         encoder_output_dim=3,
    #         encoder_hidden_dim=64,
    #
    #         fill_strategy="zero",
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #
    #         use_rel_pos=False,
    #         use_masks=True,
    #         attention_type="autoencoder",
    #
    #         masking_strategies=["random", "block"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.9,
    #     )},

    "exp71": {
        "description": "Raw KNN, autoencoder with masks, feature mixer (feat), rel_pos",
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

            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",
        )},
    "exp72": {
        "description": "Raw KNN, feature mixer (feat), space_time_attention",
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
            attention_type="space_time_attention"
        )},
    "exp73": {
        "description": "Raw KNN, feature mixer (feat), space_time_attention, masks",
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
            attention_type="space_time_attention"
        )},
    "exp74": {
        "description": "Raw KNN, feature mixer (feat+mask), space_time_attention, masks",
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
            attention_type="space_time_attention"
        )},
    "exp75": {
        "description": "Raw KNN, feature mixer (feat+mask), space_time_attention, masks, rel_pos",
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
            use_masks=True,
            attention_type="space_time_attention"
        )},
    "exp76": {
        "description": "Raw KNN, autoencoder, masks",
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
            use_masks=True,
            attention_type="autoencoder"
        )},
    "exp77": {
        "description": "Raw KNN, space_time_attention, masks",
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
            attention_type="space_time_attention"
        )},
    "exp78": {
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

            use_rel_pos=True,
            use_masks=False,
            attention_type="mha"
        )},
    "exp79": {
        "description": "Raw KNN, autoencoder, rel_pos",
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

            use_rel_pos=True,
            use_masks=False,
            attention_type="autoencoder"
        )},

    "exp80": {
        "description": "Optimal acc to ablation: Autoencoder, dynamic KNN (update every epoch, warmup, k=20), coord encoding (both) on coords, rel_pos",
        "config": ModelConfig(
            graph_mode="dynamic",
            graph_space="encoded",
            graph_metric="isotropic",

            encoder_scope="both",
            encoder_input="coords",  # Unused
            encoder_output_dim=3,
            encoder_hidden_dim=64,

            fill_strategy="zero",
            feature_mixer=False,
            feature_mixer_input="feat_mask",  # Unused

            use_rel_pos=True,
            use_masks=False,
            attention_type="autoencoder",

            n_neighbours=20,
            graph_update_frequency=1,
            graph_warmup=20,
            graph_freeze_epoch=200,
        )},
    "exp81": {
        "description": "Raw KNN, feature mixer (feat), space_time_attention, rel_pos",
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

            use_rel_pos=True,
            use_masks=False,
            attention_type="space_time_attention"
        )},
    "exp82": {
        "description": "Raw KNN, space_time_attention, rel_pos",
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
            attention_type="space_time_attention"
        )},
    "exp83": {
        "description": "Raw KNN, space_time_attention, feature mixer (feat+mask), rel_pos",
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
            attention_type="space_time_attention"
        )},
    "exp84": {
        "description": "Raw KNN, autoencoder with masks, feature mixer (feat), rel_pos, scale time to [0, 0.5]",
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

            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",
        )},
    "exp85": {
        "description": "Raw KNN, autoencoder with masks, feature mixer (feat), rel_pos, standardscaler",
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

            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",
        )},
    "exp86": {
        "description": "Raw KNN, autoencoder with masks, feature mixer (feat), rel_pos, times weighted by data availability",
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

            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",
        )},
    "exp87": {
        "description": "Raw KNN, weighted space-time-depth attention",
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
            attention_type="weighted_space_time_depth_attention"
        )},
    "exp88": {
        "description": "Raw KNN, autoencoder with masks, feature mixer (feat), rel_pos, residual gated global context",
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

            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            global_context=True,
        )},
    "exp89": {
        "description": "Raw KNN, feature mixer (feat), space_time_attention, masks, rel_pos",
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

            use_rel_pos=True,
            use_masks=True,
            attention_type="space_time_attention"
        )},
    "exp90": {
        "description": "Raw KNN, feature mixer (feat), time_space_attention, masks",
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
            attention_type="time_space_attention"
        )},

    # Per-sample masking
    "exp91": {
        "description": "Exp73 - mask_ratio=0.4",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.4
        )},
    "exp92": {
        "description": "Exp73 - mask_ratio=0.7",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.7
        )},
    "exp93": {
        "description": "Exp73 - mask_ratio=0.5",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.5
        )},
    "exp94": {
        "description": "Exp73 - mask_ratio=0.9",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.9
        )},

    # ---------------------------------------------------------------------------------------------------------------- #
    # Per-feature masking
    "exp95": {
        "description": "Exp73 - mask_ratio=0.1, masking: per_feature",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.1,
            masking_strategies=["per_feature"],
        )},
    "exp96": {
        "description": "Exp73 - mask_ratio=0.2, masking: per_feature",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.2,
            masking_strategies=["per_feature"],
        )},
    "exp97": {
        "description": "Exp73 - mask_ratio=0.3, masking: per_feature",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["per_feature"],
        )},
    "exp98": {
        "description": "Exp73 - mask_ratio=0.4, masking: per_feature",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.4,
            masking_strategies=["per_feature"],
        )},
    "exp99": {
        "description": "Exp73 - mask_ratio=0.5, masking: per_feature",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.5,
            masking_strategies=["per_feature"],
        )},
    "exp100": {
        "description": "Exp73 - mask_ratio=0.6, masking: per_feature",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.6,
            masking_strategies=["per_feature"],
        )},
    "exp101": {
        "description": "Exp73 - mask_ratio=0.7, masking: per_feature",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.7,
            masking_strategies=["per_feature"],
        )},
    "exp102": {
        "description": "Exp73 - mask_ratio=0.8, masking: per_feature",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.8,
            masking_strategies=["per_feature"],
        )},
    "exp103": {
        "description": "Exp73 - mask_ratio=0.9, masking: per_feature",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.9,
            masking_strategies=["per_feature"],
        )},

    # --------------------------------------------------------------------------------------------------------------- #
    # Spherical masking (per_sample with mask_ratio=0.3)
    "exp104": {
        "description": "Exp73 - mask_ratio=0.3, masking: sphere (radius=0.1)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.1,
        )},
    "exp105": {
        "description": "Exp73 - mask_ratio=0.3, masking: sphere (radius=0.3)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.3,
        )},
    "exp106": {
        "description": "Exp73 - mask_ratio=0.3, masking: sphere (radius=0.5)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.5,
        )},
    "exp107": {
        "description": "Exp73 - mask_ratio=0.3, masking: sphere (radius=0.7)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.7,
        )},
    "exp108": {
        "description": "Exp73 - mask_ratio=0.3, masking: sphere (radius=0.9)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.9,
        )},

    # Transect masking (various mask ratios)
    "exp109": {
        "description": "Exp73 - mask_ratio=0.3, masking: transect (width=0.05)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05
        )},
    # "exp110": {
    #     "description": "Exp73 - mask_ratio=0.4, masking: transect (width=0.05)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.4,
    #         masking_strategies=["transect"],
    #         transect_mask_p=1.0,
    #         transect_mask_width=0.05
    #     )},
    # "exp111": {
    #     "description": "Exp73 - mask_ratio=0.5, masking: transect (width=0.05)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.5,
    #         masking_strategies=["transect"],
    #         transect_mask_p=1.0,
    #         transect_mask_width=0.05
    #     )},
    # "exp112": {
    #     "description": "Exp73 - mask_ratio=0.7, masking: transect (width=0.05)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.7,
    #         masking_strategies=["transect"],
    #         transect_mask_p=1.0,
    #         transect_mask_width=0.05
    #     )},
    # "exp113": {
    #     "description": "Exp73 - mask_ratio=0.9, masking: transect (width=0.05)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.9,
    #         masking_strategies=["transect"],
    #         transect_mask_p=1.0,
    #         transect_mask_width=0.05
    #     )},

    # Transect masking (various mask ratios) - horizontal/vertical orientation
    "exp114": {
        "description": "Exp73 - mask_ratio=0.3, masking: transect (width=0.05, orientation=1.0)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
            transect_mask_orientation=1.0
        )},
    # "exp115": {
    #     "description": "Exp73 - mask_ratio=0.4, masking: transect (width=0.05, orientation=1.0)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.4,
    #         masking_strategies=["transect"],
    #         transect_mask_p=1.0,
    #         transect_mask_width=0.05,
    #         transect_mask_orientation=1.0
    #     )},
    # "exp116": {
    #     "description": "Exp73 - mask_ratio=0.5, masking: transect (width=0.05, orientation=1.0)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.5,
    #         masking_strategies=["transect"],
    #         transect_mask_p=1.0,
    #         transect_mask_width=0.05,
    #         transect_mask_orientation=1.0
    #     )},
    # "exp117": {
    #     "description": "Exp73 - mask_ratio=0.7, masking: transect (width=0.05, orientation=1.0)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.7,
    #         masking_strategies=["transect"],
    #         transect_mask_p=1.0,
    #         transect_mask_width=0.05,
    #         transect_mask_orientation=1.0
    #     )},
    # "exp118": {
    #     "description": "Exp73 - mask_ratio=0.9, masking: transect (width=0.05, orientation=1.0)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.9,
    #         masking_strategies=["transect"],
    #         transect_mask_p=1.0,
    #         transect_mask_width=0.05,
    #         transect_mask_orientation=1.0
    #     )},

    # Spherical masking (per_sample with mask_ratio=0.4)
    # "exp119": {
    #     "description": "Exp73 - mask_ratio=0.4, masking: sphere (radius=0.1)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.4,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.1,
    #     )},
    # "exp120": {
    #     "description": "Exp73 - mask_ratio=0.4, masking: sphere (radius=0.3)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.4,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.3,
    #     )},
    # "exp121": {
    #     "description": "Exp73 - mask_ratio=0.4, masking: sphere (radius=0.5)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.4,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.5,
    #     )},
    # "exp122": {
    #     "description": "Exp73 - mask_ratio=0.4, masking: sphere (radius=0.7)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.4,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.7,
    #     )},
    # "exp123": {
    #     "description": "Exp73 - mask_ratio=0.4, masking: sphere (radius=0.9)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.4,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.9,
    #     )},
    #
    # # Spherical masking (per_sample with mask_ratio=0.5)
    # "exp124": {
    #     "description": "Exp73 - mask_ratio=0.5, masking: sphere (radius=0.1)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.5,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.1,
    #     )},
    # "exp125": {
    #     "description": "Exp73 - mask_ratio=0.5, masking: sphere (radius=0.3)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.5,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.3,
    #     )},
    # "exp126": {
    #     "description": "Exp73 - mask_ratio=0.5, masking: sphere (radius=0.5)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.5,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.5,
    #     )},
    # "exp127": {
    #     "description": "Exp73 - mask_ratio=0.5, masking: sphere (radius=0.7)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.5,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.7,
    #     )},
    # "exp128": {
    #     "description": "Exp73 - mask_ratio=0.5, masking: sphere (radius=0.9)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.5,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.9,
    #     )},
    #
    # # Spherical masking (per_sample with mask_ratio=0.7)
    # "exp129": {
    #     "description": "Exp73 - mask_ratio=0.7, masking: sphere (radius=0.1)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.7,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.1,
    #     )},
    # "exp130": {
    #     "description": "Exp73 - mask_ratio=0.7, masking: sphere (radius=0.3)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.7,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.3,
    #     )},
    # "exp131": {
    #     "description": "Exp73 - mask_ratio=0.7, masking: sphere (radius=0.5)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.7,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.5,
    #     )},
    # "exp132": {
    #     "description": "Exp73 - mask_ratio=0.7, masking: sphere (radius=0.7)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.7,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.7,
    #     )},
    # "exp133": {
    #     "description": "Exp73 - mask_ratio=0.7, masking: sphere (radius=0.9)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.7,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.9,
    #     )},
    #
    # # Spherical masking (per_sample with mask_ratio=0.9)
    # "exp134": {
    #     "description": "Exp73 - mask_ratio=0.9, masking: sphere (radius=0.1)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.9,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.1,
    #     )},
    # "exp135": {
    #     "description": "Exp73 - mask_ratio=0.9, masking: sphere (radius=0.3)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.9,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.3,
    #     )},
    # "exp136": {
    #     "description": "Exp73 - mask_ratio=0.9, masking: sphere (radius=0.5)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.9,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.5,
    #     )},
    # "exp137": {
    #     "description": "Exp73 - mask_ratio=0.9, masking: sphere (radius=0.7)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.9,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.7,
    #     )},
    # "exp138": {
    #     "description": "Exp73 - mask_ratio=0.9, masking: sphere (radius=0.9)",
    #     "config": ModelConfig(
    #         feature_mixer=True,
    #         feature_mixer_input="feat",
    #         use_masks=True,
    #         attention_type="space_time_attention",
    #
    #         mask_ratio=0.9,
    #         masking_strategies=["sphere"],
    #         sphere_mask_p=1.0,
    #         sphere_mask_radius=0.9,
    #     )},

    # --------------------------------------------------------------------------------------------------------------- #
    # Random + transect masking (various mask ratios)
    "exp139": {
        "description": "Exp73 - mask_ratio=0.3, masking: random, transect (width=0.05)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05
        )},
    "exp140": {
        "description": "Exp73 - mask_ratio=0.4, masking: random, transect (width=0.05)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.4,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05
        )},
    "exp141": {
        "description": "Exp73 - mask_ratio=0.5, masking: random, transect (width=0.05)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.5,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05
        )},
    "exp142": {
        "description": "Exp73 - mask_ratio=0.7, masking: random, transect (width=0.05)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.7,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05
        )},
    "exp143": {
        "description": "Exp73 - mask_ratio=0.9, masking: random, transect (width=0.05)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.9,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05
        )},

    # Transect masking (various mask ratios) - horizontal/vertical orientation
    "exp144": {
        "description": "Exp73 - mask_ratio=0.3, masking: random, transect (width=0.05, orientation=1.0)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
            transect_mask_orientation=1.0
        )},
    "exp145": {
        "description": "Exp73 - mask_ratio=0.4, masking: random, transect (width=0.05, orientation=1.0)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.4,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
            transect_mask_orientation=1.0
        )},
    "exp146": {
        "description": "Exp73 - mask_ratio=0.5, masking: random, transect (width=0.05, orientation=1.0)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.5,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
            transect_mask_orientation=1.0
        )},
    "exp147": {
        "description": "Exp73 - mask_ratio=0.7, masking: random, transect (width=0.05, orientation=1.0)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.7,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
            transect_mask_orientation=1.0
        )},
    "exp148": {
        "description": "Exp73 - mask_ratio=0.9, masking: random, transect (width=0.05, orientation=1.0)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.9,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
            transect_mask_orientation=1.0
        )},

    # --------------------------------------------------------------------------------------------------------------- #
    # Random + spherical masking (per_sample with mask_ratio=0.3)
    "exp149": {
        "description": "Exp73 - mask_ratio=0.3, masking: random, sphere (radius=0.1)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.1,
        )},
    "exp150": {
        "description": "Exp73 - mask_ratio=0.3, masking: random, sphere (radius=0.3)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.3,
        )},
    "exp151": {
        "description": "Exp73 - mask_ratio=0.3, masking: random, sphere (radius=0.5)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.5,
        )},
    "exp152": {
        "description": "Exp73 - mask_ratio=0.3, masking: random, sphere (radius=0.7)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.7,
        )},
    "exp153": {
        "description": "Exp73 - mask_ratio=0.3, masking: random, sphere (radius=0.9)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.9,
        )},

    # Spherical masking (per_sample with mask_ratio=0.4)
    "exp154": {
        "description": "Exp73 - mask_ratio=0.4, masking: random, sphere (radius=0.1)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.4,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.1,
        )},
    "exp155": {
        "description": "Exp73 - mask_ratio=0.4, masking: random, sphere (radius=0.3)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.4,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.3,
        )},
    "exp156": {
        "description": "Exp73 - mask_ratio=0.4, masking: random, sphere (radius=0.5)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.4,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.5,
        )},
    "exp157": {
        "description": "Exp73 - mask_ratio=0.4, masking: random, sphere (radius=0.7)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.4,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.7,
        )},
    "exp158": {
        "description": "Exp73 - mask_ratio=0.4, masking: random, sphere (radius=0.9)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.4,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.9,
        )},

    # Spherical masking (per_sample with mask_ratio=0.5)
    "exp159": {
        "description": "Exp73 - mask_ratio=0.5, masking: random, sphere (radius=0.1)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.5,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.1,
        )},
    "exp160": {
        "description": "Exp73 - mask_ratio=0.5, masking: random, sphere (radius=0.3)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.5,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.3,
        )},
    "exp161": {
        "description": "Exp73 - mask_ratio=0.5, masking: random, sphere (radius=0.5)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.5,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.5,
        )},
    "exp162": {
        "description": "Exp73 - mask_ratio=0.5, masking: random, sphere (radius=0.7)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.5,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.7,
        )},
    "exp163": {
        "description": "Exp73 - mask_ratio=0.5, masking: random, sphere (radius=0.9)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.5,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.9,
        )},

    # Spherical masking (per_sample with mask_ratio=0.7)
    "exp164": {
        "description": "Exp73 - mask_ratio=0.7, masking: random, sphere (radius=0.1)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.7,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.1,
        )},
    "exp165": {
        "description": "Exp73 - mask_ratio=0.7, masking: random, sphere (radius=0.3)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.7,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.3,
        )},
    "exp166": {
        "description": "Exp73 - mask_ratio=0.7, masking: random, sphere (radius=0.5)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.7,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.5,
        )},
    "exp167": {
        "description": "Exp73 - mask_ratio=0.7, masking: random, sphere (radius=0.7)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.7,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.7,
        )},
    "exp168": {
        "description": "Exp73 - mask_ratio=0.7, masking: random, sphere (radius=0.9)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.7,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.9,
        )},

    # Spherical masking (per_sample with mask_ratio=0.9)
    "exp169": {
        "description": "Exp73 - mask_ratio=0.9, masking: random, sphere (radius=0.1)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.9,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.1,
        )},
    "exp170": {
        "description": "Exp73 - mask_ratio=0.9, masking: random, sphere (radius=0.3)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.9,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.3,
        )},
    "exp171": {
        "description": "Exp73 - mask_ratio=0.9, masking: random, sphere (radius=0.5)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.9,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.5,
        )},
    "exp172": {
        "description": "Exp73 - mask_ratio=0.9, masking: random, sphere (radius=0.7)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.9,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.7,
        )},
    "exp173": {
        "description": "Exp73 - mask_ratio=0.9, masking: random, sphere (radius=0.9)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.9,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.9,
        )},

    # Transect masking (random, w=0.02)
    "exp174": {
        "description": "Exp73 - mask_ratio=0.3, masking: transect (width=0.02, orientation=1.0)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            masking_strategies=["transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.02,
        )},
    "exp175": {
        "description": "Exp73 - mask_ratio=0.3, masking: random, transect (width=0.02, orientation=1.0)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.02,
        )},
    "exp176": {
        "description": "Exp73 - mask_ratio=0.4, masking: random, transect (width=0.02, orientation=1.0)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.4,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.02,
        )},
    "exp177": {
        "description": "Exp73 - mask_ratio=0.5, masking: random, transect (width=0.02, orientation=1.0)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.5,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.02,
        )},
    "exp178": {
        "description": "Exp73 - mask_ratio=0.7, masking: random, transect (width=0.02, orientation=1.0)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.7,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.02,
        )},
    "exp179": {
        "description": "Exp73 - mask_ratio=0.9, masking: random, transect (width=0.02, orientation=1.0)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.9,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.02,
        )},
    "exp180": {
        "description": "Exp73 - mask_ratio=0.9, masking: random, transect (width=0.02, orientation=1.0)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            mask_ratio=0.3,
            masking_strategies=["random", "transect", "sphere"],
            transect_mask_p=1.0,
            transect_mask_width=0.02,
            sphere_mask_p=1.0,
            sphere_mask_radius=0.1,
        )},

    "exp181": {
        "description": "Exp73 - beta NLL loss",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_masks=True,
            attention_type="space_time_attention",

            loss_name="beta_nll"
        )},

    ###################################################################################################################
    # --- Exp71 masking ablation
    ###################################################################################################################

    # Per-sample masking -------------------------------------------------------------------------------------------- #
    "exp182": {
        "description": "Exp71 - per-sample (mask_ratio=0.3) (=exp71)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.3,
            masking_strategies=["random"],
        )},
    "exp183": {
        "description": "Exp71 - per-sample (mask_ratio=0.4)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.4,
            masking_strategies=["random"],
        )},
    "exp184": {
        "description": "Exp71 - per-sample (mask_ratio=0.5)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.5,
            masking_strategies=["random"],
        )},
    "exp185": {
        "description": "Exp71 - per-sample (mask_ratio=0.7)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.7,
            masking_strategies=["random"],
        )},
    "exp186": {
        "description": "Exp71 - per-sample (mask_ratio=0.9)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.9,
            masking_strategies=["random"],
        )},

    # Per-sample masking + spherical (r=0.1) ------------------------------------------------------------------------- #
    "exp187": {
        "description": "Exp71 - Spherical (r=0.1)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            masking_strategies=["sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.1,
        )},
    "exp188": {
        "description": "Exp71 - per-sample (mask_ratio=0.3) + spherical (r=0.1)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.3,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.1,
        )},
    "exp189": {
        "description": "Exp71 - per-sample (mask_ratio=0.4) + spherical (r=0.1)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.4,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.1,
        )},
    "exp190": {
        "description": "Exp71 - per-sample (mask_ratio=0.5) + spherical (r=0.1)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.5,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.1,
        )},
    "exp191": {
        "description": "Exp71 - per-sample (mask_ratio=0.7) + spherical (r=0.1)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.7,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.1,
        )},
    "exp192": {
        "description": "Exp71 - per-sample (mask_ratio=0.9) + spherical (r=0.1)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.9,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.1,
        )},

    # Per-sample masking + spherical (r=0.3) ------------------------------------------------------------------------- #
    "exp193": {
        "description": "Exp71 - Spherical (r=0.3)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            masking_strategies=["sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.3,
        )},
    "exp194": {
        "description": "Exp71 - per-sample (mask_ratio=0.3) + spherical (r=0.3)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.3,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.3,
        )},
    "exp195": {
        "description": "Exp71 - per-sample (mask_ratio=0.4) + spherical (r=0.3)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.4,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.3,
        )},
    "exp196": {
        "description": "Exp71 - per-sample (mask_ratio=0.5) + spherical (r=0.3)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.5,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.3,
        )},
    "exp197": {
        "description": "Exp71 - per-sample (mask_ratio=0.7) + spherical (r=0.3)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.7,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.3,
        )},
    "exp198": {
        "description": "Exp71 - per-sample (mask_ratio=0.9) + spherical (r=0.3)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.9,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.3,
        )},

    # Per-sample masking + spherical (r=0.5) ------------------------------------------------------------------------- #
    "exp199": {
        "description": "Exp71 - Spherical (r=0.5)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            masking_strategies=["sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.5,
        )},
    "exp200": {
        "description": "Exp71 - per-sample (mask_ratio=0.3) + spherical (r=0.5)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.3,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.5,
        )},
    "exp201": {
        "description": "Exp71 - per-sample (mask_ratio=0.4) + spherical (r=0.5)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.4,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.5,
        )},
    "exp202": {
        "description": "Exp71 - per-sample (mask_ratio=0.5) + spherical (r=0.5)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.5,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.5,
        )},
    "exp203": {
        "description": "Exp71 - per-sample (mask_ratio=0.7) + spherical (r=0.5)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.7,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.5,
        )},
    "exp204": {
        "description": "Exp71 - per-sample (mask_ratio=0.9) + spherical (r=0.5)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.9,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.5,
        )},

    # Per-sample masking + spherical (r=0.7) ------------------------------------------------------------------------- #
    "exp205": {
        "description": "Exp71 - Spherical (r=0.7)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            masking_strategies=["sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.7,
        )},
    "exp206": {
        "description": "Exp71 - per-sample (mask_ratio=0.3) + spherical (r=0.7)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.3,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.7,
        )},
    "exp207": {
        "description": "Exp71 - per-sample (mask_ratio=0.4) + spherical (r=0.7)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.4,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.7,
        )},
    "exp208": {
        "description": "Exp71 - per-sample (mask_ratio=0.5) + spherical (r=0.7)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.5,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.7,
        )},
    "exp209": {
        "description": "Exp71 - per-sample (mask_ratio=0.7) + spherical (r=0.7)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.7,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.7,
        )},
    "exp210": {
        "description": "Exp71 - per-sample (mask_ratio=0.9) + spherical (r=0.7)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.9,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.7,
        )},

    # Per-sample masking + spherical (r=0.9) ------------------------------------------------------------------------- #
    "exp211": {
        "description": "Exp71 - Spherical (r=0.9)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            masking_strategies=["sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.9,
        )},
    "exp212": {
        "description": "Exp71 - per-sample (mask_ratio=0.3) + spherical (r=0.9)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.3,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.9,
        )},
    "exp213": {
        "description": "Exp71 - per-sample (mask_ratio=0.4) + spherical (r=0.9)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.4,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.9,
        )},
    "exp214": {
        "description": "Exp71 - per-sample (mask_ratio=0.5) + spherical (r=0.9)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.5,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.9,
        )},
    "exp215": {
        "description": "Exp71 - per-sample (mask_ratio=0.7) + spherical (r=0.9)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.7,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.9,
        )},
    "exp216": {
        "description": "Exp71 - per-sample (mask_ratio=0.9) + spherical (r=0.9)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.9,
            masking_strategies=["random", "sphere"],
            sphere_mask_p=1.0,
            sphere_mask_radius=0.9,
        )},

    # Per-sample masking + transect (w=0.02) ------------------------------------------------------------------------- #
    "exp217": {
        "description": "Exp71 - Transect (w=0.02)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            masking_strategies=["transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.02,
        )},
    "exp218": {
        "description": "Exp71 - per-sample (mask_ratio=0.3) + transect (w=0.02)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.3,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.02,
        )},
    "exp219": {
        "description": "Exp71 - per-sample (mask_ratio=0.4) + transect (w=0.02)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.4,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.02,
        )},
    "exp220": {
        "description": "Exp71 - per-sample (mask_ratio=0.5) + transect (w=0.02)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.5,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.02,
        )},
    "exp221": {
        "description": "Exp71 - per-sample (mask_ratio=0.7) + transect (w=0.02)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.7,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.02,
        )},
    "exp222": {
        "description": "Exp71 - per-sample (mask_ratio=0.9) + transect (w=0.02)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.9,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.02,
        )},

    # Per-sample masking + transect (w=0.05) ------------------------------------------------------------------------- #
    "exp223": {
        "description": "Exp71 - Transect (w=0.05)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            masking_strategies=["transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
        )},
    "exp224": {
        "description": "Exp71 - per-sample (mask_ratio=0.3) + transect (w=0.05)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.3,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
        )},
    "exp225": {
        "description": "Exp71 - per-sample (mask_ratio=0.4) + transect (w=0.05)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.4,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
        )},
    "exp226": {
        "description": "Exp71 - per-sample (mask_ratio=0.5) + transect (w=0.05)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.5,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
        )},
    "exp227": {
        "description": "Exp71 - per-sample (mask_ratio=0.7) + transect (w=0.05)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.7,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
        )},
    "exp228": {
        "description": "Exp71 - per-sample (mask_ratio=0.9) + transect (w=0.05)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.9,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
        )},

    # Per-sample masking + transect (w=0.05, aligned) ------------------------------------------------------------------------- #
    "exp229": {
        "description": "Exp71 - Transect (w=0.05, aligned))",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            masking_strategies=["transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
            transect_mask_orientation=1.0,
        )},
    "exp230": {
        "description": "Exp71 - per-sample (mask_ratio=0.3) + transect (w=0.05, aligned))",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.3,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
            transect_mask_orientation=1.0,
        )},
    "exp231": {
        "description": "Exp71 - per-sample (mask_ratio=0.4) + transect (w=0.05, aligned))",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.4,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
            transect_mask_orientation=1.0,
        )},
    "exp232": {
        "description": "Exp71 - per-sample (mask_ratio=0.5) + transect (w=0.05, aligned))",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.5,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
            transect_mask_orientation=1.0,
        )},
    "exp233": {
        "description": "Exp71 - per-sample (mask_ratio=0.7) + transect (w=0.05, aligned))",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.7,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
            transect_mask_orientation=1.0,
        )},
    "exp234": {
        "description": "Exp71 - per-sample (mask_ratio=0.9) + transect (w=0.05, aligned))",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.9,
            masking_strategies=["random", "transect"],
            transect_mask_p=1.0,
            transect_mask_width=0.05,
            transect_mask_orientation=1.0,
        )},

    # Per-feature masking ------------------------------------------------------------------------------------------- #
    "exp235": {
        "description": "Exp71 - Per-feature masking (r=0.1)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.1,
            masking_strategies=["per_feature"],
        )},
    "exp236": {
        "description": "Exp71 - Per-feature masking (r=0.2)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.2,
            masking_strategies=["per_feature"],
        )},
    "exp237": {
        "description": "Exp71 - Per-feature masking (r=0.3)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.3,
            masking_strategies=["per_feature"],
        )},
    "exp238": {
        "description": "Exp71 - Per-feature masking (r=0.4)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.4,
            masking_strategies=["per_feature"],
        )},
    "exp239": {
        "description": "Exp71 - Per-feature masking (r=0.5)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.5,
            masking_strategies=["per_feature"],
        )},
    "exp240": {
        "description": "Exp71 - Per-feature masking (r=0.6)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.6,
            masking_strategies=["per_feature"],
        )},
    "exp241": {
        "description": "Exp71 - Per-feature masking (r=0.7)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.7,
            masking_strategies=["per_feature"],
        )},
    "exp242": {
        "description": "Exp71 - Per-feature masking (r=0.8)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.8,
            masking_strategies=["per_feature"],
        )},
    "exp243": {
        "description": "Exp71 - Per-feature masking (r=0.9)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.9,
            masking_strategies=["per_feature"],
        )},

    # Combined masking ---------------------------------------------------------------------------------------------- #
    "exp244": {
        "description": "Exp71 - Combination: per-sample (mask_ratio=0.3), spherical (r=0.1), transect (w=0.02)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            use_rel_pos=True,
            use_masks=True,
            attention_type="autoencoder",

            mask_ratio=0.3,
            masking_strategies=["random", "transect", "sphere"],
            transect_mask_p=1.0,
            transect_mask_width=0.02,
            sphere_mask_p=1.0,
            sphere_mask_radius=0.1,
        )},

    #################################################
    # --- Architectures
    ################################################

    # Time-space attention
    "exp245": {
        "description": "Time_space_attention",
        "config": ModelConfig(
            feature_mixer=False,
            attention_type="time_space_attention"
        )},
    "exp246": {
        "description": "Time_space_attention, feature mixer (feat)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            attention_type="time_space_attention"
        )},
    "exp247": {
        "description": "Time_space_attention, feature mixer (feat+mask)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat_mask",
            attention_type="time_space_attention"
        )},

    # Space-time-depth attention
    # Same as exp18
    # "exp248": {
    #     "description": "Space_time_depth_attention",
    #     "config": ModelConfig(
    #         attention_type="space_time_depth_attention"
    #     )},
    "exp248": {
        "description": "Space_time_depth_attention, feature mixer (feat)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            attention_type="space_time_depth_attention"
        )},
    "exp249": {
        "description": "Space_time_depth_attention, feature mixer (feat+mask)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat_mask",
            attention_type="space_time_depth_attention"
        )},

    # Encoder layer
    # Same as exp10
    # "exp250": {
    #     "description": "Transformer encoder",
    #     "config": ModelConfig(
    #         attention_type="transformer_encoder"
    #     )},
    "exp250": {
        "description": "Transformer encoder, feature mixer (feat)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat",
            attention_type="transformer_encoder"
        )},
    "exp251": {
        "description": "Transformer encoder, feature mixer (feat+mask)",
        "config": ModelConfig(
            feature_mixer=True,
            feature_mixer_input="feat_mask",
            attention_type="transformer_encoder"
        )},

}
