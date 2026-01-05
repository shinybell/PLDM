"""量子化モジュール (FSQ, VQ-VAE)

FSQ: Finite Scalar Quantization
参考: https://arxiv.org/abs/2309.15505
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class QuantizerOutput:
    """量子化器の出力"""
    quantized: torch.Tensor  # 量子化後の値 (B, T, num_codebooks)
    indices: torch.Tensor  # 離散インデックス (B, T, num_codebooks)
    commitment_loss: Optional[torch.Tensor] = None  # VQ-VAE用
    codebook_loss: Optional[torch.Tensor] = None  # VQ-VAE用


class FSQ(nn.Module):
    """Finite Scalar Quantization (FSQ)

    各次元を固定された離散レベルに量子化する。
    VQ-VAEと異なり、コードブックの学習が不要で、コードブック崩壊も発生しない。

    Args:
        levels: List[int] - 各次元の離散レベル数 (例: [8, 8, 8, 5, 5, 5])
        dim: int - 入力次元数
        num_codebooks: int - FSQのコードブック次元（通常はlen(levels)）
        project_in: bool - 入力をnum_codebooks次元に射影するか
        eps: float - 数値安定性のための小さな値

    Example:
        >>> fsq = FSQ(levels=[8, 5, 5], dim=512, num_codebooks=3)
        >>> x = torch.randn(2, 10, 512)  # (B, T, dim)
        >>> output = fsq(x)
        >>> print(output.quantized.shape)  # (2, 10, 3)
        >>> print(output.indices.shape)    # (2, 10, 3)
    """

    def __init__(
        self,
        levels: List[int],
        dim: int,
        num_codebooks: Optional[int] = None,
        project_in: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()

        if num_codebooks is None:
            num_codebooks = len(levels)

        self.levels = levels
        self.num_codebooks = num_codebooks
        self.dim = dim
        self.eps = eps

        # コードブックサイズ（全次元の積）
        self.codebook_size = math.prod(levels)

        # 入力を num_codebooks 次元に射影
        if project_in:
            self.project_in = nn.Linear(dim, num_codebooks)
        else:
            assert dim == num_codebooks, \
                f"dim ({dim}) must equal num_codebooks ({num_codebooks}) when project_in=False"
            self.project_in = nn.Identity()

        # 各次元の量子化境界を事前計算
        # levels[i] 個のレベルを [-1, 1] の範囲に均等配置
        # register_bufferはTensorのリストを直接扱えないので、個別に登録
        for i, level in enumerate(self.levels):
            bound = torch.linspace(-1, 1, level)
            self.register_buffer(f'_boundary_{i}', bound)

        self.register_buffer('_levels_tensor', torch.tensor(levels))

    def _get_boundary(self, dim_idx: int) -> torch.Tensor:
        """指定次元の量子化境界を取得"""
        return getattr(self, f'_boundary_{dim_idx}')

    def _quantize_per_dim(
        self,
        z: torch.Tensor,
        dim_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """1次元の量子化

        Args:
            z: (*, ) - 量子化する値
            dim_idx: int - 次元インデックス

        Returns:
            quantized: (*, ) - 量子化後の値
            indices: (*, ) - インデックス (0 ~ levels[dim_idx]-1)
        """
        boundaries = self._get_boundary(dim_idx).to(z.device)

        # 最も近い境界を見つける
        # z: (B, T), boundaries: (L,) -> distances: (B, T, L)
        distances = torch.abs(z.unsqueeze(-1) - boundaries)
        indices = distances.argmin(dim=-1)  # (B, T)

        # インデックスから量子化値を取得
        quantized = boundaries[indices]  # (B, T)

        return quantized, indices

    def forward(self, z: torch.Tensor) -> QuantizerOutput:
        """
        Args:
            z: (B, T, dim) - 入力（連続表現）

        Returns:
            QuantizerOutput:
                quantized: (B, T, num_codebooks) - 量子化後の値
                indices: (B, T, num_codebooks) - 離散インデックス
        """
        original_shape = z.shape[:-1]  # (B, T)

        # 1. 入力を num_codebooks 次元に射影
        z_proj = self.project_in(z)  # (B, T, num_codebooks)

        # 2. tanh で [-1, 1] に正規化
        z_normalized = torch.tanh(z_proj)

        # 3. 各次元を独立に量子化
        quantized_list = []
        indices_list = []

        for i in range(self.num_codebooks):
            z_i = z_normalized[..., i]  # (B, T)
            q_i, idx_i = self._quantize_per_dim(z_i, i)
            quantized_list.append(q_i)
            indices_list.append(idx_i)

        quantized = torch.stack(quantized_list, dim=-1)  # (B, T, num_codebooks)
        indices = torch.stack(indices_list, dim=-1)  # (B, T, num_codebooks)

        # 4. Straight-Through Estimator (STE) で勾配を流す
        # Forward: 量子化値を使用
        # Backward: 勾配はそのまま流す
        quantized = z_normalized + (quantized - z_normalized).detach()

        return QuantizerOutput(
            quantized=quantized,
            indices=indices,
        )

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """インデックスから量子化値に変換

        Args:
            indices: (B, T, num_codebooks) - 離散インデックス

        Returns:
            codes: (B, T, num_codebooks) - 量子化値
        """
        codes_list = []
        for i in range(self.num_codebooks):
            boundaries = self._boundaries[i].to(indices.device)
            codes_i = boundaries[indices[..., i]]  # (B, T)
            codes_list.append(codes_i)

        codes = torch.stack(codes_list, dim=-1)  # (B, T, num_codebooks)
        return codes

    def get_codebook_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """各次元のインデックスを単一のコードブックインデックスに変換

        Args:
            indices: (B, T, num_codebooks) - 各次元のインデックス

        Returns:
            codebook_indices: (B, T) - 結合されたインデックス (0 ~ codebook_size-1)
        """
        # Mixed-radix encoding
        # 例: levels=[8, 5, 5] の場合
        # index = idx[0] * (5*5) + idx[1] * 5 + idx[2]

        multipliers = [1]
        for level in reversed(self.levels[1:]):
            multipliers.insert(0, multipliers[0] * level)

        multipliers = torch.tensor(multipliers, device=indices.device)
        codebook_indices = (indices * multipliers).sum(dim=-1)  # (B, T)

        return codebook_indices

    def codebook_indices_to_indices(self, codebook_indices: torch.Tensor) -> torch.Tensor:
        """単一のコードブックインデックスを各次元のインデックスに変換

        Args:
            codebook_indices: (B, T) - 結合されたインデックス

        Returns:
            indices: (B, T, num_codebooks) - 各次元のインデックス
        """
        indices_list = []
        remaining = codebook_indices

        for i, level in enumerate(self.levels):
            # 混合基数デコード
            if i < len(self.levels) - 1:
                # 後続の次元の積
                divisor = math.prod(self.levels[i+1:])
                idx_i = remaining // divisor
                remaining = remaining % divisor
            else:
                idx_i = remaining

            indices_list.append(idx_i)

        indices = torch.stack(indices_list, dim=-1)  # (B, T, num_codebooks)
        return indices


class FSQEmbedding(nn.Module):
    """FSQインデックスをembedding空間に変換

    量子化されたインデックスを元の表現次元に戻すためのembedding層

    Args:
        levels: List[int] - 各次元のレベル数
        embed_dim: int - 出力embedding次元
    """

    def __init__(self, levels: List[int], embed_dim: int):
        super().__init__()

        self.levels = levels
        self.num_codebooks = len(levels)
        self.embed_dim = embed_dim

        # 方法1: 各次元のインデックスを個別にembedして結合
        self.embeddings = nn.ModuleList([
            nn.Embedding(level, embed_dim // self.num_codebooks)
            for level in levels
        ])

        # 最終的な次元調整（必要に応じて）
        total_embed_dim = sum(embed_dim // self.num_codebooks for _ in levels)
        if total_embed_dim != embed_dim:
            self.project_out = nn.Linear(total_embed_dim, embed_dim)
        else:
            self.project_out = nn.Identity()

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: (B, T, num_codebooks) - 離散インデックス

        Returns:
            embeddings: (B, T, embed_dim) - embedding表現
        """
        # 各次元のembeddingを取得
        embeds = []
        for i in range(self.num_codebooks):
            embed_i = self.embeddings[i](indices[..., i])  # (B, T, embed_dim//C)
            embeds.append(embed_i)

        # 結合
        combined = torch.cat(embeds, dim=-1)  # (B, T, total_embed_dim)

        # 次元調整
        output = self.project_out(combined)  # (B, T, embed_dim)

        return output


# ユーティリティ関数
def compute_codebook_usage(indices: torch.Tensor, levels: List[int]) -> dict:
    """コードブック利用率を計算

    Args:
        indices: (B, T, num_codebooks) - 離散インデックス
        levels: List[int] - 各次元のレベル数

    Returns:
        dict: 利用率の統計
    """
    stats = {}

    # 全体の形状を平坦化
    indices_flat = indices.reshape(-1, indices.shape[-1])  # (N, C)

    # 各次元ごとの利用率
    per_dim_usage = []
    for i, level in enumerate(levels):
        unique_codes = torch.unique(indices_flat[:, i])
        usage = len(unique_codes) / level
        per_dim_usage.append(usage)

    stats['per_dim_usage'] = per_dim_usage
    stats['avg_usage'] = sum(per_dim_usage) / len(per_dim_usage)

    # 全コードブックの利用率（組み合わせ）
    # これは計算コストが高いので、サンプリングして計算
    sample_size = min(10000, len(indices_flat))
    sampled_indices = indices_flat[:sample_size]

    # 各次元のインデックスをタプルに変換して unique を取る
    unique_combinations = set()
    for idx in sampled_indices:
        combination = tuple(idx.cpu().tolist())
        unique_combinations.add(combination)

    total_combinations = 1
    for level in levels:
        total_combinations *= level

    stats['combination_usage'] = len(unique_combinations) / min(sample_size, total_combinations)
    stats['unique_combinations'] = len(unique_combinations)
    stats['total_combinations'] = total_combinations

    return stats
