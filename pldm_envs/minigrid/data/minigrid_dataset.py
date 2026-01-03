"""
MiniGrid環境のデータセットクラス

オフラインデータからMiniGrid LongHorizon環境のデータをロードし、
PLDM学習用に前処理を行います。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, NamedTuple
from pathlib import Path
from dataclasses import dataclass

from pldm_envs.utils.normalizer import Normalizer

# MiniGridのアクション数（7つの離散アクション）
MINIGRID_NUM_ACTIONS = 7


class MiniGridSample(NamedTuple):
    """
    MiniGridデータセットのサンプル

    Attributes:
        states: 観測画像 [T, C, H, W] torch.Tensor (float32)
        actions: アクション [T-1, NUM_ACTIONS] torch.Tensor (float32) - one-hot
        locations: エージェント位置 [T, 2] torch.Tensor (float32) - (x, y)座標
        rewards: 報酬 [T-1] torch.Tensor (float32) - オプション
        dones: 終端フラグ [T-1] torch.Tensor (bool) - オプション
    """
    states: torch.Tensor
    actions: torch.Tensor
    locations: torch.Tensor
    rewards: Optional[torch.Tensor] = None
    dones: Optional[torch.Tensor] = None


@dataclass
class MiniGridDatasetConfig:
    """
    MiniGridデータセットの設定

    Attributes:
        data_path: データファイルのパス (.npz)
        val_path: 検証データのパス (.npz, オプション)
        sample_length: サンプルの時系列長（コンテキスト長）
        img_size: 画像サイズ（64, 72など）
        normalize_images: 画像を[0, 1]に正規化するか
        include_rewards: 報酬を含めるか
        include_dones: 終端フラグを含めるか
        batch_size: バッチサイズ
        crop_length: データセットを切り詰める長さ（None=全て使用）
        train: 訓練モードか検証モードか
        quick_debug: デバッグモード（少量データで高速化）
    """
    data_path: str = "data/minigrid/level1_train.npz"  # デフォルトパス
    val_path: Optional[str] = None  # 検証データパス（オプション）
    sample_length: int = 16
    img_size: int = 64
    normalize_images: bool = True
    include_rewards: bool = False
    include_dones: bool = False
    batch_size: int = 32
    crop_length: Optional[int] = None
    train: bool = True
    quick_debug: bool = False


class MiniGridDataset(torch.utils.data.Dataset):
    """
    MiniGrid LongHorizon環境のデータセットクラス

    オフラインデータ（.npzファイル）からデータをロードし、
    スライディングウィンドウでサンプリングします。

    データフォーマット:
        NPZ形式:
            - observations: [N_episodes, T, H, W, C] uint8 [0-255]
            - actions: [N_episodes, T-1] int
            - rewards: [N_episodes, T-1] float (オプション)
            - dones: [N_episodes, T-1] bool (オプション)

    使用例:
        config = MiniGridDatasetConfig(
            data_path='data/minigrid/level1_train.npz',
            sample_length=16,
            img_size=64,
        )
        dataset = MiniGridDataset(config)
        sample = dataset[0]  # MiniGridSample
    """

    def __init__(
        self,
        config: MiniGridDatasetConfig,
        normalizer: Optional[Normalizer] = None,
    ):
        """
        Args:
            config: データセット設定
            normalizer: 正規化器（訓練データから作成）
        """
        self.config = config
        self.normalizer = normalizer

        # データのロード
        self._load_data()

        print(f"Loaded MiniGrid dataset from {config.data_path}")
        print(f"  Total samples: {len(self)}")

    def _load_data(self):
        """NPZファイルまたはディレクトリからデータをロード"""
        data_path = Path(self.config.data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        print(f"Loading data from {data_path}")

        if data_path.is_dir():
            # ディレクトリからロード (.npy files)
            print("Loading from directory (mmap_mode='r')")
            
            def load_npy(filename):
                return np.load(data_path / filename, mmap_mode='r')

            self.observations = load_npy('observations.npy')
            self.actions = load_npy('actions.npy')
            
            if (data_path / 'positions.npy').exists():
                self.locations = load_npy('positions.npy')
            elif (data_path / 'locations.npy').exists():
                self.locations = load_npy('locations.npy')
            else:
                 raise ValueError(
                    "Dataset must contain 'positions.npy' or 'locations.npy'. "
                    "Please regenerate the dataset with position information."
                )

            # 報酬（オプション）
            if (data_path / 'rewards.npy').exists() and self.config.include_rewards:
                self.rewards = load_npy('rewards.npy')
            else:
                self.rewards = None

            # 終端フラグ（オプション）
            if (data_path / 'dones.npy').exists() and self.config.include_dones:
                self.dones = load_npy('dones.npy')
            else:
                self.dones = None

        elif data_path.suffix == '.npz':
            # NPZファイルからロード
            print("Loading from .npz file")

            # メモリマップモードで読み込み（大量データでもメモリ節約）
            if self.config.quick_debug:
                data = np.load(data_path, allow_pickle=True)
            else:
                data = np.load(data_path, allow_pickle=True, mmap_mode='r')

            # observations: [N_episodes, T, H, W, C] uint8
            self.observations = data['observations']

            # actions: [N_episodes, T-1] int
            self.actions = data['actions']

            # locations: [N_episodes, T, 2] float32 - エージェント位置 (x, y)
            if 'positions' in data:
                self.locations = data['positions']
            elif 'locations' in data:
                self.locations = data['locations']
            else:
                raise ValueError(
                    "Dataset must contain 'positions' or 'locations' field. "
                    "Please regenerate the dataset with position information."
                )

            # 報酬（オプション）
            if 'rewards' in data and self.config.include_rewards:
                self.rewards = data['rewards']
            else:
                self.rewards = None

            # 終端フラグ（オプション）
            if 'dones' in data and self.config.include_dones:
                self.dones = data['dones']
            else:
                self.dones = None
        
        else:
             raise ValueError(
                f"Unsupported file format: {data_path.suffix}. "
                "Only .npz format or directory of .npy files is supported."
            )

        print(f"  Observations: {self.observations.shape} {self.observations.dtype}")
        print(f"  Actions: {self.actions.shape} {self.actions.dtype}")
        print(f"  Locations: {self.locations.shape} {self.locations.dtype}")

        # スライディングウィンドウの設定
        self._setup_slicing()

    def _setup_slicing(self):
        """スライディングウィンドウの設定"""
        # observations.dtype が object の場合（可変長エピソード）
        if self.observations.dtype == object:
            # 可変長エピソード
            self.variable_length = True
            self._setup_variable_length_slicing()
        else:
            # 固定長エピソード
            self.variable_length = False
            self._setup_fixed_length_slicing()

    def _setup_fixed_length_slicing(self):
        """固定長エピソードのスライシング設定"""
        # エピソード長
        self.episode_length = self.observations.shape[1]

        # サンプル長がエピソード長より長い場合は調整
        if self.config.sample_length > self.episode_length:
            print(
                f"Warning: sample_length ({self.config.sample_length}) > "
                f"episode_length ({self.episode_length}). "
                f"Using episode_length."
            )
            self.effective_sample_length = self.episode_length
        else:
            self.effective_sample_length = self.config.sample_length

        # 各エピソードから取れるスライス数
        self.slices_per_episode = (
            self.episode_length - self.effective_sample_length + 1
        )

        # 総スライス数
        self.total_slices = len(self.observations) * self.slices_per_episode

        print(f"  Episode length: {self.episode_length}")
        print(f"  Sample length: {self.effective_sample_length}")
        print(f"  Slices per episode: {self.slices_per_episode}")
        print(f"  Total slices: {self.total_slices}")

    def _setup_variable_length_slicing(self):
        """可変長エピソードのスライシング設定"""
        # 各エピソードの長さとスライス数を計算
        self.episode_lengths = []
        self.slices_per_episode_list = []
        cumulative_slices = 0
        self.slice_to_episode_map = []

        for ep_idx in range(len(self.observations)):
            ep_length = len(self.observations[ep_idx])
            self.episode_lengths.append(ep_length)

            # このエピソードから取れるスライス数
            effective_sample_length = min(self.config.sample_length, ep_length)
            slices = max(1, ep_length - effective_sample_length + 1)
            self.slices_per_episode_list.append(slices)

            # スライスIDからエピソードへのマッピング
            for _ in range(slices):
                self.slice_to_episode_map.append((ep_idx, cumulative_slices))
                cumulative_slices += 1

        self.total_slices = cumulative_slices

        print(f"  Variable length episodes: {len(self.observations)}")
        print(f"  Episode lengths: min={min(self.episode_lengths)}, "
              f"max={max(self.episode_lengths)}, "
              f"mean={np.mean(self.episode_lengths):.1f}")
        print(f"  Sample length: {self.config.sample_length}")
        print(f"  Total slices: {self.total_slices}")

    def __len__(self):
        """データセットの長さを返す"""
        if self.config.crop_length is not None:
            return min(self.config.crop_length, self.total_slices)
        return self.total_slices

    def __getitem__(self, idx: int) -> MiniGridSample:
        """
        インデックスに対応するサンプルを返す

        Args:
            idx: サンプルのインデックス

        Returns:
            MiniGridSample: 観測、アクション、その他の情報を含む
        """
        if self.variable_length:
            return self._getitem_variable_length(idx)
        else:
            return self._getitem_fixed_length(idx)

    def _getitem_fixed_length(self, idx: int) -> MiniGridSample:
        """固定長エピソードからサンプルを取得"""
        # エピソードとスライス位置を計算
        episode_idx = idx // self.slices_per_episode
        slice_start = idx % self.slices_per_episode
        slice_end = slice_start + self.effective_sample_length

        # データの取得
        obs = self.observations[episode_idx, slice_start:slice_end]  # [T, H, W, C]
        actions = self.actions[episode_idx, slice_start:slice_end-1]  # [T-1]
        locations = self.locations[episode_idx, slice_start:slice_end]  # [T, 2]

        # 画像の前処理
        obs = self._preprocess_observations(obs)

        # Tensorに変換
        states = torch.from_numpy(obs).float()  # [T, C, H, W]
        actions_indices = torch.from_numpy(actions).long()  # [T-1]
        # One-hotエンコーディング: [T-1] -> [T-1, NUM_ACTIONS]
        actions = F.one_hot(actions_indices, num_classes=MINIGRID_NUM_ACTIONS).float()
        locations = torch.from_numpy(np.array(locations)).float()  # [T, 2]

        # 報酬の取得（オプション）
        if self.rewards is not None:
            rewards = torch.from_numpy(
                self.rewards[episode_idx, slice_start:slice_end-1]
            ).float()
        else:
            rewards = None

        # 終端フラグの取得（オプション）
        if self.dones is not None:
            dones = torch.from_numpy(
                self.dones[episode_idx, slice_start:slice_end-1]
            ).bool()
        else:
            dones = None

        sample = MiniGridSample(
            states=states,
            actions=actions,
            locations=locations,
            rewards=rewards,
            dones=dones,
        )

        return sample

    def _getitem_variable_length(self, idx: int) -> MiniGridSample:
        """可変長エピソードからサンプルを取得"""
        # エピソードとスライス位置を特定
        episode_idx = None
        for ep_idx, slices in enumerate(self.slices_per_episode_list):
            if idx < slices:
                episode_idx = ep_idx
                slice_offset = idx
                break
            idx -= slices

        if episode_idx is None:
            raise IndexError(f"Index {idx} out of range")

        ep_length = self.episode_lengths[episode_idx]
        effective_sample_length = min(self.config.sample_length, ep_length)

        slice_start = slice_offset
        slice_end = slice_start + effective_sample_length

        # データの取得 (object配列から取り出す)
        obs = self.observations[episode_idx][slice_start:slice_end]  # [T, H, W, C]
        actions = self.actions[episode_idx][slice_start:slice_end-1]  # [T-1]
        locations = self.locations[episode_idx][slice_start:slice_end]  # [T, 2]

        # numpy配列に変換（明示的にdtypeを指定）
        obs = np.array(obs, dtype=np.uint8)
        actions = np.array(actions, dtype=np.int64)
        locations = np.array(locations, dtype=np.float32)

        # 画像の前処理
        obs = self._preprocess_observations(obs)

        # Tensorに変換
        states = torch.from_numpy(obs).float()  # [T, C, H, W]
        actions_indices = torch.from_numpy(actions).long()  # [T-1]
        # One-hotエンコーディング: [T-1] -> [T-1, NUM_ACTIONS]
        actions = F.one_hot(actions_indices, num_classes=MINIGRID_NUM_ACTIONS).float()
        locations = torch.from_numpy(locations).float()  # [T, 2]

        # 報酬とdones（オプション）
        rewards = None
        dones = None
        if self.rewards is not None:
            rewards_data = np.array(self.rewards[episode_idx][slice_start:slice_end-1])
            rewards = torch.from_numpy(rewards_data).float()
        if self.dones is not None:
            dones_data = np.array(self.dones[episode_idx][slice_start:slice_end-1])
            dones = torch.from_numpy(dones_data).bool()

        sample = MiniGridSample(
            states=states,
            actions=actions,
            locations=locations,
            rewards=rewards,
            dones=dones,
        )

        return sample

    def _preprocess_observations(self, obs: np.ndarray) -> np.ndarray:
        """
        観測画像の前処理

        Args:
            obs: [T, H, W, C] uint8 [0-255]

        Returns:
            preprocessed: [T, C, H, W] float32 [0-1] or uint8 [0-255]
        """
        # チャンネルを先頭に移動: [T, H, W, C] -> [T, C, H, W]
        obs = np.transpose(obs, (0, 3, 1, 2))

        # リサイズ（必要な場合）
        current_size = obs.shape[2]
        if current_size != self.config.img_size:
            obs = self._resize_observations(obs)

        # 正規化 (0-255 -> 0-1)
        if self.config.normalize_images:
            obs = obs.astype(np.float32) / 255.0

        return obs

    def _resize_observations(self, obs: np.ndarray) -> np.ndarray:
        """
        観測画像をリサイズ

        Args:
            obs: [T, C, H, W]

        Returns:
            resized: [T, C, img_size, img_size]
        """
        # torch.nn.functionalを使用してリサイズ
        import torch.nn.functional as F

        obs_tensor = torch.from_numpy(obs).float()
        resized = F.interpolate(
            obs_tensor,
            size=(self.config.img_size, self.config.img_size),
            mode='bilinear',
            align_corners=False
        )

        # 元のdtypeに戻す
        if obs.dtype == np.uint8:
            return resized.numpy().astype(np.uint8)
        else:
            return resized.numpy()
