"""MiniGrid Environment Construction for Evaluation"""
import gymnasium as gym
import numpy as np
from typing import List, Optional
from minigrid.wrappers import RGBImgObsWrapper

from pldm_envs.minigrid.wrappers import ResizeObservationWrapper
from pldm_envs.minigrid.data.minigrid_dataset import MiniGridDatasetConfig
from pldm_envs.utils.normalizer import Normalizer


def construct_eval_envs(
    seed: int,
    minigrid_config: MiniGridDatasetConfig,
    n_envs: int = 4,
    level: str = "level1",
    normalizer: Optional[Normalizer] = None,
) -> List[gym.Env]:
    """
    MiniGrid評価用の環境を複数作成

    Args:
        seed: ランダムシード
        minigrid_config: MiniGridデータセット設定
        n_envs: 作成する環境の数
        level: 難易度レベル ('level1', 'level2', 'level3')
        normalizer: 正規化器（オプション）

    Returns:
        環境のリスト
    """
    # レベルに応じた環境名を決定
    level_to_env = {
        "level1": "MiniGrid-LongHorizon-Level1-v0",
        "level2": "MiniGrid-LongHorizon-Level2-v0",
        "level3": "MiniGrid-LongHorizon-Level3-v0",
    }

    if level not in level_to_env:
        raise ValueError(
            f"Unknown level: {level}. Must be one of {list(level_to_env.keys())}"
        )

    env_name = level_to_env[level]

    envs = []
    for i in range(n_envs):
        # 各環境に異なるシードを設定
        env_seed = seed + i if seed is not None else None

        # 環境を作成
        env = gym.make(env_name)

        # RGB観測ラッパーを適用
        env = RGBImgObsWrapper(env)

        # リサイズラッパーを適用（必要な場合）
        if minigrid_config.img_size != 64:
            env = ResizeObservationWrapper(env, size=minigrid_config.img_size)

        # シードを設定
        if env_seed is not None:
            env.reset(seed=env_seed)

        envs.append(env)

    return envs
