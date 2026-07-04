
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"


@dataclass
class PathsConfig:
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    data_graphs: str = "data/graphs"
    data_embeddings: str = "data/embeddings"
    checkpoints: str = "checkpoints"
    mlruns: str = "mlruns"
    drugbank_xml: str = "data/raw/drugbank_full_database.xml"


@dataclass
class GINConfig:
    num_layers: int = 3
    hidden_dim: int = 128
    embedding_dim: int = 64
    dropout: float = 0.15
    eps: float = 0.0
    train_eps: bool = True


@dataclass
class ContrastiveConfig:
    temperature: float = 0.07
    batch_size: int = 64
    epochs: int = 300
    lr: float = 0.001
    weight_decay: float = 1e-5
    warmup_epochs: int = 10
    projection_dim: int = 32
    aug_atom_mask_ratio: float = 0.15
    aug_atom_mask_ratio_2: float = 0.20
    aug_bond_drop_ratio: float = 0.10
    aug_noise_sigma: float = 0.05


@dataclass
class RGCNConfig:
    num_layers: int = 2
    hidden_dim: int = 128
    dropout: float = 0.2
    num_interaction_types: int = 86
    num_severity_levels: int = 4
    num_bases: int = 16


@dataclass
class LossWeightsConfig:
    lambda_binary: float = 1.0
    lambda_severity: float = 0.8
    lambda_type: float = 0.5
    lambda_faers: float = 0.3


@dataclass
class FinetuneConfig:
    epochs: int = 200
    lr: float = 5e-4
    weight_decay: float = 1e-5
    batch_size: int = 512
    freeze_gin_epochs: int = 30
    grad_accumulation_steps: int = 4
    early_stopping_patience: int = 20
    eval_every: int = 5


@dataclass
class ScoringThresholds:
    safe: int = 30
    review: int = 60
    high_risk: int = 80


@dataclass
class ScoringWeights:
    contraindicated: float = 2.0
    major: float = 1.5
    moderate: float = 1.0
    minor: float = 0.3


@dataclass
class ScoringConfig:
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    thresholds: ScoringThresholds = field(default_factory=ScoringThresholds)
    mc_dropout_samples: int = 50
    low_confidence_threshold: float = 0.25


@dataclass
class RecommenderConfig:
    top_k_candidates: int = 20
    top_k_return: int = 3
    atc_match_required: bool = False


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_drugs_per_request: int = 15
    fuzzy_match_threshold: int = 70
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])


@dataclass
class HardwareConfig:
    mixed_precision: bool = True
    cudnn_benchmark: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    gradient_clip_norm: float = 1.0


@dataclass
class MLflowConfig:
    experiment_name: str = "medsafe_training"
    tracking_uri: str = "mlruns"


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    gin: GINConfig = field(default_factory=GINConfig)
    gin_full: GINConfig = field(default_factory=lambda: GINConfig(num_layers=5, hidden_dim=256, embedding_dim=128))
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    rgcn: RGCNConfig = field(default_factory=RGCNConfig)
    rgcn_full: RGCNConfig = field(default_factory=lambda: RGCNConfig(num_layers=3, hidden_dim=256, num_bases=32))
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    recommender: RecommenderConfig = field(default_factory=RecommenderConfig)
    api: APIConfig = field(default_factory=APIConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    severity_mapping: dict[str, int] = field(default_factory=dict)
    cyp450_enzymes: list[str] = field(default_factory=lambda: ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"])
    qt_prolonging_drugs: list[str] = field(default_factory=list)
    cns_depressant_categories: list[str] = field(default_factory=list)
    nsaid_drugs: list[str] = field(default_factory=list)
    anticoagulant_drugs: list[str] = field(default_factory=list)
    dataset: dict[str, Any] = field(default_factory=dict)


def _dict_to_dataclass(cls: type, data: dict) -> Any:
    """Recursively convert a dict to a dataclass instance."""
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            val = data[f.name]
            # Try to recurse for nested dataclasses
            try:
                ftype = f.type
                if isinstance(ftype, str):
                    ftype = eval(ftype)
                import dataclasses as dc
                if dc.is_dataclass(ftype) and isinstance(val, dict):
                    val = _dict_to_dataclass(ftype, val)
            except Exception:
                pass
            kwargs[f.name] = val
    return cls(**kwargs)


def load_config(config_path: Path = CONFIG_PATH, full_mode: bool = False) -> Config:
    """Load config.yaml and return a typed Config object."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = Config()

    # Paths
    if "paths" in raw:
        cfg.paths = PathsConfig(**{k: v for k, v in raw["paths"].items() if hasattr(PathsConfig, k)})

    # GIN config (default or full)
    gin_key = "gin_full" if full_mode else "gin"
    if gin_key in raw:
        cfg.gin = GINConfig(**{k: v for k, v in raw[gin_key].items() if hasattr(GINConfig, k)})

    # Contrastive config (merge full overrides if applicable)
    if "contrastive" in raw:
        cfg.contrastive = ContrastiveConfig(**{k: v for k, v in raw["contrastive"].items() if hasattr(ContrastiveConfig, k)})
    if full_mode and "contrastive_full" in raw:
        for k, v in raw["contrastive_full"].items():
            if hasattr(cfg.contrastive, k):
                setattr(cfg.contrastive, k, v)

    # R-GCN config
    rgcn_key = "rgcn_full" if full_mode else "rgcn"
    if rgcn_key in raw:
        cfg.rgcn = RGCNConfig(**{k: v for k, v in raw[rgcn_key].items() if hasattr(RGCNConfig, k)})

    # Loss weights
    if "loss_weights" in raw:
        cfg.loss_weights = LossWeightsConfig(**{k: v for k, v in raw["loss_weights"].items()})

    # Finetune
    if "finetune" in raw:
        finetune_raw = raw["finetune"].copy()
        if full_mode and "finetune_full" in raw:
            finetune_raw.update(raw["finetune_full"])
        cfg.finetune = FinetuneConfig(**{k: v for k, v in finetune_raw.items() if hasattr(FinetuneConfig, k)})

    # Scoring
    if "scoring" in raw:
        sc = raw["scoring"]
        weights = ScoringWeights(**sc.get("weights", {}))
        thresholds = ScoringThresholds(**{k: v for k, v in sc.get("thresholds", {}).items() if hasattr(ScoringThresholds, k)})
        cfg.scoring = ScoringConfig(
            weights=weights,
            thresholds=thresholds,
            mc_dropout_samples=sc.get("mc_dropout_samples", 50),
            low_confidence_threshold=sc.get("low_confidence_threshold", 0.25),
        )

    # API
    if "api" in raw:
        cfg.api = APIConfig(**{k: v for k, v in raw["api"].items() if hasattr(APIConfig, k)})

    # Hardware
    if "hardware" in raw:
        cfg.hardware = HardwareConfig(**{k: v for k, v in raw["hardware"].items() if hasattr(HardwareConfig, k)})

    # MLflow
    if "mlflow" in raw:
        cfg.mlflow = MLflowConfig(**{k: v for k, v in raw["mlflow"].items() if hasattr(MLflowConfig, k)})

    # Recommender
    if "recommender" in raw:
        cfg.recommender = RecommenderConfig(**{k: v for k, v in raw["recommender"].items() if hasattr(RecommenderConfig, k)})

    # Plain dicts / lists
    cfg.severity_mapping = raw.get("severity_mapping", {})
    cfg.cyp450_enzymes = raw.get("cyp450_enzymes", ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"])
    cfg.qt_prolonging_drugs = raw.get("qt_prolonging_drugs", [])
    cfg.cns_depressant_categories = raw.get("cns_depressant_categories", [])
    cfg.nsaid_drugs = raw.get("nsaid_drugs", [])
    cfg.anticoagulant_drugs = raw.get("anticoagulant_drugs", [])
    cfg.dataset = raw.get("dataset", {})

    return cfg


if __name__ == "__main__":
    cfg = load_config()
    print(f"Config loaded. GIN layers: {cfg.gin.num_layers}, hidden: {cfg.gin.hidden_dim}")
    cfg_full = load_config(full_mode=True)
    print(f"Full config. GIN layers: {cfg_full.gin.num_layers}, hidden: {cfg_full.gin.hidden_dim}")
