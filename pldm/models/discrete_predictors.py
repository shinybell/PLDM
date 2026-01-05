"""離散表現用のPredictor

FSQで量子化された離散表現を予測するためのモジュール
"""

import torch
import torch.nn as nn
from typing import List, Optional
from dataclasses import dataclass

from pldm.models.enums import PredictorOutput


@dataclass
class DiscretePredictorOutput:
    """離散Predictor出力"""
    predictions: torch.Tensor  # (T, B, num_codebooks, max_levels) - logits
    indices: torch.Tensor  # (T, B, num_codebooks) - 予測インデックス
    hidden_states: Optional[torch.Tensor] = None  # RNNの隠れ状態

    # 既存PredictorOutputとの互換性のため
    obs_component: Optional[torch.Tensor] = None
    propio_component: Optional[torch.Tensor] = None
    prior_mus: Optional[torch.Tensor] = None
    prior_vars: Optional[torch.Tensor] = None
    prior_logits: Optional[torch.Tensor] = None
    priors: Optional[torch.Tensor] = None
    posterior_mus: Optional[torch.Tensor] = None
    posterior_vars: Optional[torch.Tensor] = None
    posterior_logits: Optional[torch.Tensor] = None
    posteriors: Optional[torch.Tensor] = None


class DiscreteRNNPredictor(nn.Module):
    """離散表現用RNN Predictor

    Re-embeddingされた離散表現とアクションから、次の離散インデックスを予測する。
    各コードブック次元ごとに分類問題として定式化。

    Args:
        config: PredictorConfig
        repr_dim: int - Re-embedding次元
        action_dim: int - アクション次元
        num_codebooks: int - FSQコードブック次元数
        num_levels: List[int] - 各次元のクラス数
        rnn_hidden_dim: int - RNNの隠れ層次元
        rnn_layers: int - RNN層数
        predictor_ln: bool - Layer Normalizationを使うか
        backbone_ln: Optional[nn.Module] - Backboneと共有するLN
    """

    def __init__(
        self,
        config,
        repr_dim: int,
        action_dim: int,
        num_codebooks: int,
        num_levels: List[int],
        rnn_hidden_dim: int = 512,
        rnn_layers: int = 1,
        predictor_ln: bool = True,
        backbone_ln: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.config = config
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.num_codebooks = num_codebooks
        self.num_levels = num_levels
        self.rnn_hidden_dim = rnn_hidden_dim

        # 既存のPredictorとの互換性のため
        self.pred_propio_dim = 0  # DiscreteJEPAは固有受容状態を予測しない

        # Layer Normalization
        if hasattr(config, 'tie_backbone_ln') and config.tie_backbone_ln and backbone_ln is not None:
            self.final_ln = backbone_ln
        elif predictor_ln:
            self.final_ln = nn.LayerNorm(repr_dim)
        else:
            self.final_ln = nn.Identity()

        # アクションエンコーダ（オプション）
        if hasattr(config, 'action_encoder_arch') and config.action_encoder_arch:
            # アクションをエンコード
            from pldm.models.misc import build_mlp
            action_dims = [int(d) for d in config.action_encoder_arch.split('-')]
            self.action_encoder = build_mlp([action_dim] + action_dims)
            encoded_action_dim = action_dims[-1]
        else:
            self.action_encoder = nn.Identity()
            encoded_action_dim = action_dim

        # RNN本体
        rnn_input_dim = repr_dim + encoded_action_dim
        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_layers,
            batch_first=False,  # (T, B, D) format
        )

        # 各コードブック次元ごとに分類ヘッド
        self.prediction_heads = nn.ModuleList([
            nn.Linear(rnn_hidden_dim, num_levels[i])
            for i in range(num_codebooks)
        ])

        # Residual connection用の射影（オプション）
        if hasattr(config, 'residual') and config.residual:
            self.residual_proj = nn.Linear(repr_dim, rnn_hidden_dim)
        else:
            self.residual_proj = None

    def forward(
        self,
        z_discrete: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ):
        """1ステップの予測

        Args:
            z_discrete: (B, repr_dim) - 現在の離散表現（re-embedded）
            action: (B, action_dim) - アクション
            hidden: (num_layers, B, hidden_dim) - RNN隠れ状態

        Returns:
            logits: (B, num_codebooks, max_levels) - 各次元のlogits
            indices: (B, num_codebooks) - 予測インデックス
            hidden: (num_layers, B, hidden_dim) - 更新された隠れ状態
        """
        B = z_discrete.shape[0]

        # Layer Normalization
        z_discrete = self.final_ln(z_discrete)

        # アクションエンコード
        action_encoded = self.action_encoder(action)

        # RNN入力
        rnn_input = torch.cat([z_discrete, action_encoded], dim=-1)  # (B, D)
        rnn_input = rnn_input.unsqueeze(0)  # (1, B, D)

        # RNN forward
        rnn_out, hidden = self.rnn(rnn_input, hidden)
        rnn_out = rnn_out.squeeze(0)  # (B, hidden_dim)

        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(z_discrete)
            rnn_out = rnn_out + residual

        # 各コードブック次元の予測
        logits_list = []
        indices_list = []
        max_levels = max(self.num_levels)

        for i, head in enumerate(self.prediction_heads):
            logit = head(rnn_out)  # (B, num_levels[i])
            logits_list.append(logit)
            indices_list.append(logit.argmax(dim=-1))  # (B,)

        # パディングして統一形状にする
        padded_logits = torch.zeros(
            B, self.num_codebooks, max_levels,
            device=z_discrete.device,
            dtype=z_discrete.dtype
        )
        for i, logit in enumerate(logits_list):
            padded_logits[:, i, :self.num_levels[i]] = logit

        indices = torch.stack(indices_list, dim=-1)  # (B, num_codebooks)

        return padded_logits, indices, hidden

    def forward_multiple(
        self,
        state_encs: torch.Tensor,
        actions: torch.Tensor,
        T: int,
        fsq_embedding: Optional[nn.Module] = None,
        teacher_forcing_ratio: float = 0.0,
        ground_truth_indices: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """複数ステップの予測

        Args:
            state_encs: (1, B, repr_dim) - 初期状態（re-embedded）
            actions: (T, B, action_dim) - アクション系列
            T: int - 予測ステップ数
            fsq_embedding: Optional[nn.Module] - FSQEmbedding層（自己回帰用）
            teacher_forcing_ratio: float - Teacher forcingの割合
            ground_truth_indices: Optional[(T+1, B, num_codebooks)] - 正解インデックス

        Returns:
            DiscretePredictorOutput
        """
        B = state_encs.shape[1]
        device = state_encs.device

        # 初期状態
        current_state = state_encs.squeeze(0)  # (B, repr_dim)
        hidden = None

        all_logits = []
        all_indices = []
        all_embeddings = []  # Probing用のre-embedded表現

        for t in range(T):
            # 1ステップ予測
            logits_t, indices_t, hidden = self.forward(
                current_state, actions[t], hidden
            )

            all_logits.append(logits_t)
            all_indices.append(indices_t)

            # 次のステップの入力を決定
            # Teacher forcing vs 自己回帰
            use_teacher_forcing = (
                teacher_forcing_ratio > 0 and
                ground_truth_indices is not None and
                torch.rand(1).item() < teacher_forcing_ratio
            )

            if use_teacher_forcing:
                # Ground truthを使用
                next_indices = ground_truth_indices[t + 1]  # (B, num_codebooks)
            else:
                # 予測を使用
                next_indices = indices_t

            # Re-embed（次のステップの入力）
            if fsq_embedding is not None:
                current_state = fsq_embedding(next_indices)  # (B, repr_dim)
                all_embeddings.append(current_state)
            else:
                # fsq_embeddingがない場合はそのまま（デバッグ用）
                # 実際の訓練では必須
                current_state = current_state  # keep current
                all_embeddings.append(current_state)

        # スタック
        predictions = torch.stack(all_logits, dim=0)  # (T, B, C, L)
        indices = torch.stack(all_indices, dim=0)  # (T, B, C)
        embeddings = torch.stack(all_embeddings, dim=0) if all_embeddings else None  # (T, B, D)

        return DiscretePredictorOutput(
            predictions=predictions,
            indices=indices,
            hidden_states=hidden,
            obs_component=embeddings,  # Probing用のre-embedded表現
        )


class DiscreteTransformerPredictor(nn.Module):
    """離散表現用Transformer Predictor（将来の拡張用）

    現在は未実装。Transformerベースの予測器を実装する場合に使用。
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("DiscreteTransformerPredictor is not implemented yet")


def build_discrete_predictor(
    config,
    repr_dim: int,
    action_dim: int,
    num_codebooks: int,
    num_levels: List[int],
    **kwargs
):
    """離散Predictorのファクトリ関数

    Args:
        config: DiscretePredictorConfig
        repr_dim: int - 表現次元
        action_dim: int - アクション次元
        num_codebooks: int - コードブック次元
        num_levels: List[int] - 各次元のレベル数

    Returns:
        DiscretePredictor
    """
    arch = config.predictor_arch

    if arch == "discrete_rnn":
        return DiscreteRNNPredictor(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            num_codebooks=num_codebooks,
            num_levels=num_levels,
            rnn_hidden_dim=config.rnn_hidden_dim,
            rnn_layers=config.rnn_layers,
            predictor_ln=config.predictor_ln,
        )
    elif arch == "discrete_transformer":
        return DiscreteTransformerPredictor(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            num_codebooks=num_codebooks,
            num_levels=num_levels,
        )
    else:
        raise ValueError(f"Unknown discrete predictor architecture: {arch}")
