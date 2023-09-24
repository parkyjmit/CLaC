from dataclasses import dataclass

@dataclass
class Paths:
    log: str
    data: str

@dataclass
class HyperParams:
    batch_size: int
    max_epochs: int
    progress_bar_refresh_rate: int
    gpus: int

@dataclass
class GraphEncoderParams:
    num_layers: int
    hidden_dim: int
    dropout: float
    num_classes: int
    num_heads: int

@dataclass
class GraphEncoder:
    model_name: str
    params: GraphEncoderParams
    
@dataclass
class TextEncoderParams:
    num_layers: int
    hidden_dim: int
    dropout: float
    num_classes: int
    num_heads: int

@dataclass
class TextEncoder:
    model_name: str
    params: TextEncoderParams

@dataclass
class CLaMPConfigs:
    paths: Paths
    hyperparams: HyperParams
    graph_encoder_params: GraphEncoderParams
    text_encoder_params: TextEncoderParams
