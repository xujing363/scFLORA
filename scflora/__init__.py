"""scFLORA: a protein-informed generative foundation model for plant single-cell transcriptomes."""

__version__ = "1.0.0"

from .config import DataConfig, LossConfig, ModelConfig, TrainConfig
from .families import FamilySpace, GenomeIndex
from .genes import GeneTable, canon_gene_id
from .inference import embed, embed_adata, load_pretrained, predict_rates, prepare_species, virtual_knockout
from .model import PointBatch, ScFLORA

__all__ = [
    "ModelConfig", "DataConfig", "LossConfig", "TrainConfig",
    "FamilySpace", "GenomeIndex", "GeneTable", "canon_gene_id",
    "ScFLORA", "PointBatch",
    "load_pretrained", "prepare_species", "embed", "embed_adata", "predict_rates", "virtual_knockout",
]
