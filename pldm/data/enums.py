from typing import Optional, NamedTuple
from enum import Enum, auto
from dataclasses import dataclass, field # <-- fieldをインポート
from pldm.configs import ConfigBase

from pldm_envs.wall.data.offline_wall import OfflineWallDatasetConfig
from pldm_envs.wall.data.wall import WallDatasetConfig
from pldm_envs.wall.data.single import DotDatasetConfig
from pldm_envs.wall.data.wall_expert import WallExpertDatasetConfig

from pldm_envs.diverse_maze.enums import D4RLDatasetConfig


class DatasetType(Enum):
    Single = auto()
    Multiple = auto()
    Wall = auto()
    WallExpert = auto()
    D4RL = auto()
    D4RLEigf = auto()
    LocoMaze = auto()


class ProbingDatasets(NamedTuple):
    ds: DatasetType
    val_ds: DatasetType
    extra_datasets: dict = {}


class Datasets(NamedTuple):
    ds: DatasetType
    val_ds: DatasetType
    probing_datasets: Optional[ProbingDatasets] = None
    l2_probing_datasets: Optional[ProbingDatasets] = None


@dataclass
class DataConfig(ConfigBase):
    dataset_type: DatasetType = DatasetType.Single
    # 🚨 修正: ミュータブルなデフォルト値を field(default_factory=...) に変更
    dot_config: DotDatasetConfig = field(default_factory=DotDatasetConfig)
    wall_config: WallDatasetConfig = field(default_factory=WallDatasetConfig)
    offline_wall_config: OfflineWallDatasetConfig = field(default_factory=OfflineWallDatasetConfig)
    wall_expert_config: WallExpertDatasetConfig = field(default_factory=WallExpertDatasetConfig)

    # if "AMD" not in torch.cuda.get_device_name(0):
    # D4RLDatasetConfig も同様に修正
    d4rl_config: D4RLDatasetConfig = field(default_factory=D4RLDatasetConfig)

    normalize: bool = False
    min_max_normalize_state: bool = False
    normalizer_hardset: bool = False
    quick_debug: bool = False
    num_workers: int = 0