"""MiniGrid Planning Utility Functions"""
import torch
import numpy as np


def determine_terminations(locations, targets, error_threshold):
    """
    各エピソードがいつ目標に到達したかを判定

    Args:
        locations: 各ステップでのエージェント位置 [T, B, 2]
        targets: 目標位置 [B, 2]
        error_threshold: 到達判定の閾値

    Returns:
        terminations: 各エピソードの終了ステップ（到達しなかった場合は最大ステップ）
    """
    terminations = []
    num_envs = len(targets)

    for i in range(num_envs):
        target = targets[i].cpu()
        terminated = False

        for t, loc in enumerate(locations):
            current_loc = loc[i].cpu()
            error = (current_loc - target).pow(2).sum().sqrt()

            if error < error_threshold:
                terminations.append(t)
                terminated = True
                break

        if not terminated:
            terminations.append(len(locations) - 1)

    return terminations


def calculate_success_rate(locations, targets, error_threshold):
    """
    ゴール到達率を計算

    Args:
        locations: 最終ステップでのエージェント位置 [B, 2]
        targets: 目標位置 [B, 2]
        error_threshold: 到達判定の閾値

    Returns:
        success_rate: 成功率 [0, 1]
    """
    errors = (locations - targets).pow(2).sum(dim=-1).sqrt()
    successes = (errors < error_threshold).float()
    success_rate = successes.mean().item()

    return success_rate


def extract_agent_position(obs):
    """
    観測から推定されるエージェント位置を抽出

    Note:
        MiniGridでは観測が部分観測のため、正確な位置情報は得られない。
        環境から直接取得する必要がある。
        この関数はプレースホルダーとして定義。

    Args:
        obs: 観測 [B, C, H, W]

    Returns:
        positions: エージェント位置の推定値 [B, 2]
    """
    # MiniGridの場合、観測からは位置を直接抽出できないため、
    # 環境から取得する必要がある
    # ここではダミーの実装
    batch_size = obs.shape[0]
    return torch.zeros(batch_size, 2)
