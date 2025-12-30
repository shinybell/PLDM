"""
MiniGrid環境のデータセットクラス

オフラインデータ、またはオンライン生成の両方に対応
"""

import torch
import numpy as np
from typing import Optional
import pickle
from pathlib import Path

from pldm_envs.minigrid.enums import MiniGridSample, MiniGridDatasetConfig


class MiniGridDataset(torch.utils.data.Dataset):
    """
    MiniGrid環境のデータセットクラス

    オフラインデータ（.npz, .pklファイル）からデータをロード

    データフォーマット:
        NPZ形式:
            - observations: [N_episodes, T, H, W, C]
            - actions: [N_episodes, T-1]
            - rewards: [N_episodes, T-1] (オプション)
            - dones: [N_episodes, T-1] (オプション)

        PKL形式:
            - List of episodes, 各エピソードは dict{'obs', 'actions', ...}
    """

    def __init__(
        self,
        config: MiniGridDatasetConfig,
        normalizer=None,
    ):
        self.config = config
        self.normalizer = normalizer

        if config.online_mode:
            raise NotImplementedError(
                "Online mode not yet implemented. "
                "Please use offline mode with data_path specified."
            )

        # データのロード
        self._load_data()

        print(f"Loaded MiniGrid {config.env_name} dataset with {len(self)} samples")

    def _load_data(self):
        """データをロードする"""
        if self.config.data_path is None:
            raise ValueError("data_path must be specified for offline mode")

        data_path = Path(self.config.data_path)

        if data_path.suffix == '.npz':
            self._load_npz(data_path)
        elif data_path.suffix == '.pkl':
            self._load_pkl(data_path)
        else:
            raise ValueError(
                f"Unsupported file format: {data_path.suffix}. "
                "Supported formats: .npz, .pkl"
            )

        # スライディングウィンドウの設定
        self._setup_slicing()

    def _load_npz(self, path: Path):
        """NPZファイルからデータをロード"""
        print(f"Loading NPZ file from {path}")

        if self.config.quick_debug:
            data = np.load(path, allow_pickle=True)
        else:
            # メモリマップモードで読み込み（メモリ節約）
            data = np.load(path, allow_pickle=True, mmap_mode='r')

        # observations: [N_episodes, T, H, W, C]
        self.observations = data['observations']

        # actions: [N_episodes, T-1]
        self.actions = data['actions']

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

        print(f"  Observations shape: {self.observations.shape}")
        print(f"  Actions shape: {self.actions.shape}")

    def _load_pkl(self, path: Path):
        """PKLファイルからデータをロード"""
        print(f"Loading PKL file from {path}")

        with open(path, 'rb') as f:
            episodes = pickle.load(f)

        # List[Dict] -> numpy arrays に変換
        self.observations = np.array([ep['obs'] for ep in episodes])
        self.actions = np.array([ep['actions'] for ep in episodes])

        if self.config.include_rewards:
            self.rewards = np.array([ep.get('rewards', None) for ep in episodes])
        else:
            self.rewards = None

        if self.config.include_dones:
            self.dones = np.array([ep.get('dones', None) for ep in episodes])
        else:
            self.dones = None

    def _setup_slicing(self):
        """スライディングウィンドウの設定"""
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
        # エピソードとスライス位置を計算
        episode_idx = idx // self.slices_per_episode
        slice_start = idx % self.slices_per_episode
        slice_end = slice_start + self.effective_sample_length

        # データの取得
        obs = self.observations[episode_idx, slice_start:slice_end]  # [T, H, W, C]
        actions = self.actions[episode_idx, slice_start:slice_end-1]  # [T-1]

        # 画像の前処理
        obs = self._preprocess_observations(obs)

        # Tensorに変換
        states = torch.from_numpy(obs).float()  # [T, C, H, W]
        actions = torch.from_numpy(actions).long().unsqueeze(-1)  # [T-1, 1]

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
            rewards=rewards,
            dones=dones,
        )

        return sample

    def _preprocess_observations(self, obs: np.ndarray) -> np.ndarray:
        """
        観測画像の前処理

        Args:
            obs: [T, H, W, C]

        Returns:
            preprocessed: [T, C, H, W]
        """
        # チャンネルを先頭に移動: [T, H, W, C] -> [T, C, H, W]
        obs = np.transpose(obs, (0, 3, 1, 2))

        # リサイズ（必要な場合）
        if obs.shape[2] != self.config.img_size or obs.shape[3] != self.config.img_size:
            obs = self._resize_observations(obs)

        # 正規化 (0-255 -> 0-1)
        if self.config.normalize_images and obs.max() > 1:
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
        try:
            import cv2
            T, C, H, W = obs.shape
            resized = np.zeros((T, C, self.config.img_size, self.config.img_size), dtype=obs.dtype)

            for t in range(T):
                for c in range(C):
                    resized[t, c] = cv2.resize(
                        obs[t, c],
                        (self.config.img_size, self.config.img_size),
                        interpolation=cv2.INTER_AREA
                    )
            return resized

        except ImportError:
            # cv2がない場合はtorchvisionを使用
            import torch
            import torch.nn.functional as F

            obs_tensor = torch.from_numpy(obs).float()
            resized = F.interpolate(
                obs_tensor,
                size=(self.config.img_size, self.config.img_size),
                mode='bilinear',
                align_corners=False
            )
            return resized.numpy()
