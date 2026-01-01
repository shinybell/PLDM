import dataclasses

from pldm_envs.wall.data.offline_wall import OfflineWallDataset
from pldm_envs.wall.data.wall import WallDataset
from pldm_envs.wall.data.single import DotDataset
from pldm_envs.wall.data.wall_expert import (
    WrappedWallExpertDataset,
)
from pldm_envs.wall.data.wall_passing_test import WallPassingTestDataset
from pldm_envs.wall.data.border_passing_test import BorderPassingTestDataset

from pldm.data.utils import make_dataloader, make_dataloader_for_prebatched_ds


# if "AMD" not in torch.cuda.get_device_name(0):
from pldm_envs.diverse_maze.d4rl import D4RLDataset

# Atari dataset
from pldm_envs.atari.data.atari_dataset import AtariDataset

# MiniGrid dataset
from pldm_envs.minigrid.data import MiniGridDataset, minigrid_collate_fn

from pldm.probing.evaluator import ProbingConfig
from pldm_envs.utils.normalizer import Normalizer
from pldm.data.enums import DataConfig, DatasetType, ProbingDatasets, Datasets


class DatasetFactory:
    def __init__(
        self,
        config: DataConfig,
        probing_cfg: ProbingConfig = ProbingConfig(),
        disable_l2: bool = True,
    ):
        self.config = config
        self.probing_cfg = probing_cfg
        self.disable_l2 = disable_l2

    def create_datasets(self):
        if self.config.dataset_type == DatasetType.Single:
            return self._create_single_datasets()
        elif self.config.dataset_type == DatasetType.Wall:
            return self._create_wall_datasets()
        elif self.config.dataset_type == DatasetType.WallExpert:
            return self._create_wall_expert_datasets()
        elif self.config.dataset_type == DatasetType.D4RL:
            return self._create_d4rl_datasets()
        elif self.config.dataset_type == DatasetType.LocoMaze:
            return self._create_locomaze_datasets()
        elif self.config.dataset_type == DatasetType.Atari:
            return self._create_atari_datasets()
        elif self.config.dataset_type == DatasetType.MiniGrid:
            return self._create_minigrid_datasets()
        else:
            raise NotImplementedError

    def _create_single_datasets(self):
        ds = DotDataset(self.config.dot_config)
        val_ds = DotDataset(
            dataclasses.replace(self.config.dot_config, train=False),
            normalizer=ds.normalizer,
        )

        datasets = Datasets(ds=ds, val_ds=val_ds)

        return datasets

    def _create_wall_datasets(self):
        if self.config.offline_wall_config.use_offline:
            ds = OfflineWallDataset(config=self.config.offline_wall_config)
            ds = make_dataloader(
                ds=ds, loader_config=self.config, suffix="offline_wall"
            )
        else:
            ds = WallDataset(self.config.wall_config)
            ds = make_dataloader_for_prebatched_ds(
                probe_ds,
                loader_config=self.config,
            )

        probing_datasets = self._create_wall_probing_datasets(ds.normalizer)

        datasets = Datasets(
            ds=ds,
            val_ds=None,
            probing_datasets=probing_datasets,
        )

        return datasets

    def _create_wall_probing_datasets(self, normalizer: Normalizer):
        probe_ds = WallDataset(
            dataclasses.replace(
                self.config.wall_config,
                size=self.config.wall_config.val_size,
                train=False,
                n_steps=self.probing_cfg.l1_depth,
                fix_wall_batch_k=None,
                expert_cross_wall_rate=0,
            )
        )
        probe_ds = make_dataloader_for_prebatched_ds(
            probe_ds,
            loader_config=self.config,
            normalizer=normalizer,
        )

        probe_val_ds = WallDataset(
            dataclasses.replace(
                self.config.wall_config,
                size=self.config.wall_config.val_size,
                n_steps=self.probing_cfg.l1_depth,
                fix_wall_batch_k=None,
                train=False,
                expert_cross_wall_rate=0,
            )
        )
        probe_val_ds = make_dataloader_for_prebatched_ds(
            probe_val_ds,
            loader_config=self.config,
            normalizer=normalizer,
        )

        extra_datasets = {}

        if self.probing_cfg.probe_wall:
            wall_test_ds = WallPassingTestDataset(
                dataclasses.replace(
                    self.config.wall_config,
                    size=self.config.wall_config.val_size,
                    n_steps=self.probing_cfg.l1_depth,
                    fix_wall_batch_k=None,
                    train=False,
                )
            )
            extra_datasets["wall_test"] = make_dataloader_for_prebatched_ds(
                wall_test_ds, loader_config=self.config, normalizer=normalizer
            )
        if self.probing_cfg.probe_border:
            border_test_ds = BorderPassingTestDataset(
                dataclasses.replace(
                    self.config.wall_config,
                    size=self.config.wall_config.val_size,
                    n_steps=self.probing_cfg.l1_depth,
                    fix_wall_batch_k=None,
                    train=False,
                )
            )
            extra_datasets["border_test"] = make_dataloader_for_prebatched_ds(
                border_test_ds, loader_config=self.config, normalizer=normalizer
            )

        probing_datasets = ProbingDatasets(
            ds=probe_ds, val_ds=probe_val_ds, extra_datasets=extra_datasets
        )

        return probing_datasets

    def _create_wall_expert_datasets(self):
        ds = WallDataset(dataclasses.replace(self.config.wall_config, train=False))
        ds = WrappedWallExpertDataset(
            self.config.wall_expert_config, normalizer=ds.normalizer
        )
        val_ds = WrappedWallExpertDataset(
            dataclasses.replace(self.config.wall_expert_config, train=False),
            normalizer=ds.normalizer,
        )

        datasets = Datasets(
            ds=ds,
            val_ds=None,
        )

        return datasets

    def _create_d4rl_datasets(self):
        ds = D4RLDataset(self.config.d4rl_config)
        ds = make_dataloader(ds=ds, loader_config=self.config)

        probe_ds = D4RLDataset(
            dataclasses.replace(
                self.config.d4rl_config,
                path=self.probing_cfg.train_path,
                images_path=self.probing_cfg.train_images_path,
                sample_length=self.probing_cfg.l1_depth,
            ),
        )
        probe_ds = make_dataloader(
            ds=probe_ds,
            loader_config=self.config,
            normalizer=ds.normalizer,
            suffix="probe_train",
        )

        probe_val_ds = D4RLDataset(
            dataclasses.replace(
                self.config.d4rl_config,
                path=self.probing_cfg.val_path,
                images_path=self.probing_cfg.val_images_path,
                sample_length=self.probing_cfg.l1_depth,
                train=False,
                crop_length=50000,
                batch_size=64,
            ),
        )

        probe_val_ds = make_dataloader(
            ds=probe_val_ds,
            loader_config=self.config,
            normalizer=ds.normalizer,
            suffix="probe_val",
        )

        datasets = Datasets(
            ds=ds,
            val_ds=None,
            probing_datasets=ProbingDatasets(ds=probe_ds, val_ds=probe_val_ds),
        )

        return datasets

    def _create_atari_datasets(self):
        """
        Atariデータセットを作成

        Returns:
            Datasets: 訓練データセット、検証データセット（オプション）
        """
        # 訓練データセット
        ds = AtariDataset(self.config.atari_config)
        ds = make_dataloader(
            ds=ds,
            loader_config=self.config,
            suffix="atari_train"
        )

        # 検証データセット（オプション）
        val_ds = None
        if self.config.atari_config.val_path is not None:
            val_ds = AtariDataset(
                dataclasses.replace(
                    self.config.atari_config,
                    data_path=self.config.atari_config.val_path,
                    train=False,
                )
            )
            val_ds = make_dataloader(
                ds=val_ds,
                loader_config=self.config,
                normalizer=ds.normalizer,
                suffix="atari_val",
                train=False,
            )

        datasets = Datasets(
            ds=ds,
            val_ds=val_ds,
        )

        return datasets

    def _create_minigrid_datasets(self):
        """
        MiniGridデータセットを作成

        Returns:
            Datasets: 訓練データセット、検証データセット、Probingデータセット（オプション）
        """
        # 訓練データセット
        ds = MiniGridDataset(self.config.minigrid_config)
        ds = make_dataloader(
            ds=ds,
            loader_config=self.config,
            suffix="minigrid_train",
            collate_fn=minigrid_collate_fn,
        )

        # 検証データセット（オプション）
        val_ds = None
        if self.config.minigrid_config.val_path is not None:
            val_ds = MiniGridDataset(
                dataclasses.replace(
                    self.config.minigrid_config,
                    data_path=self.config.minigrid_config.val_path,
                    train=False,
                )
            )
            val_ds = make_dataloader(
                ds=val_ds,
                loader_config=self.config,
                normalizer=ds.normalizer,
                suffix="minigrid_val",
                train=False,
                collate_fn=minigrid_collate_fn,
            )

        # Probingデータセット（検証データが存在し、probing設定が有効な場合）
        probing_datasets = None
        if (
            self.config.minigrid_config.val_path is not None
            and self.probing_cfg is not None
        ):
            probing_datasets = self._create_minigrid_probing_datasets(
                normalizer=ds.normalizer
            )

        datasets = Datasets(
            ds=ds,
            val_ds=val_ds,
            probing_datasets=probing_datasets,
        )

        return datasets

    def _create_minigrid_probing_datasets(self, normalizer: Normalizer):
        """
        MiniGrid用のProbingデータセットを作成

        Args:
            normalizer: 訓練データセットから作成されたNormalizer

        Returns:
            ProbingDatasets: Probing用の訓練・検証データセット
        """
        # Probing訓練データセット
        probe_ds = MiniGridDataset(
            dataclasses.replace(
                self.config.minigrid_config,
                # Probingではより短いシーケンスを使用
                sample_length=self.probing_cfg.l1_depth,
                train=True,
            )
        )
        probe_ds = make_dataloader(
            ds=probe_ds,
            loader_config=self.config,
            normalizer=normalizer,
            suffix="probe_train",
            collate_fn=minigrid_collate_fn,
        )

        # Probing検証データセット
        probe_val_ds = MiniGridDataset(
            dataclasses.replace(
                self.config.minigrid_config,
                data_path=self.config.minigrid_config.val_path,
                sample_length=self.probing_cfg.l1_depth,
                train=False,
                batch_size=64,
                crop_length=50000 if self.config.minigrid_config.crop_length is None else self.config.minigrid_config.crop_length,
            ),
        )

        probe_val_ds = make_dataloader(
            ds=probe_val_ds,
            loader_config=self.config,
            normalizer=normalizer,
            suffix="probe_val",
            collate_fn=minigrid_collate_fn,
        )

        probing_datasets = ProbingDatasets(ds=probe_ds, val_ds=probe_val_ds)

        return probing_datasets
