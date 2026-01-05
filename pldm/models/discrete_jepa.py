"""DiscreteJEPA: FSQベースの離散JEPA

FSQ (Finite Scalar Quantization)を用いて潜在表現を離散化し、
離散空間で予測を行うJEPA
"""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass

from pldm.configs_discrete import DiscreteJEPAConfig
from pldm.models.encoders.encoders import build_backbone
from pldm.models.quantizers import FSQ, FSQEmbedding
from pldm.models.discrete_predictors import build_discrete_predictor


@dataclass
class DiscreteForwardResult:
    """DiscreteJEPAの出力"""
    # Backbone出力
    backbone_output: any  # BackboneOutput
    ema_backbone_output: Optional[any] = None

    # FSQ関連
    z_continuous: torch.Tensor = None  # FSQ前の連続表現
    z_quantized: torch.Tensor = None  # 量子化後の値
    z_indices: torch.Tensor = None  # 離散インデックス
    z_discrete: torch.Tensor = None  # Re-embedding後の離散表現

    # Predictor出力
    pred_output: any = None  # DiscretePredictorOutput
    actions: torch.Tensor = None


class DiscreteJEPA(nn.Module):
    """FSQベースの離散JEPA

    アーキテクチャ:
        画像 → Backbone → 連続表現 (z_continuous)
                            ↓
                        FSQ量子化 → 離散インデックス (z_indices)
                            ↓
                        Re-embedding → 離散表現 (z_discrete)
                            ↓
        Predictor (z_discrete + action → 次のz_indices)

    Args:
        config: DiscreteJEPAConfig
        input_dim: tuple - 入力画像の次元 (C, H, W)
        normalizer: Optional - データ正規化器
        use_propio_pos: bool - 固有受容位置を使うか
        use_propio_vel: bool - 固有受容速度を使うか
    """

    def __init__(
        self,
        config: DiscreteJEPAConfig,
        input_dim,
        normalizer=None,
        use_propio_pos: bool = False,
        use_propio_vel: bool = False,
    ):
        super().__init__()

        self.config = config
        self.use_propio_pos = use_propio_pos
        self.use_propio_vel = use_propio_vel
        self.normalizer = normalizer

        # 1. Backbone（既存のエンコーダを使用）
        self.backbone = build_backbone(
            config.backbone,
            input_dim=input_dim,
        )

        # Backboneの出力次元
        self.spatial_repr_dim = self.backbone.output_dim
        if isinstance(self.spatial_repr_dim, tuple):
            import operator
            from functools import reduce
            self.repr_dim = reduce(operator.mul, self.spatial_repr_dim)
        else:
            self.repr_dim = self.spatial_repr_dim

        # 2. EMA Backbone（オプション）
        if config.momentum > 0:
            self.backbone_ema = build_backbone(
                config.backbone,
                input_dim=input_dim,
            )[0]  # build_backboneはタプルを返す場合がある
            self.backbone_ema.load_state_dict(self.backbone.state_dict())
            for param in self.backbone_ema.parameters():
                param.requires_grad = False
        else:
            self.backbone_ema = None

        # 3. FSQ量子化層
        if config.use_fsq:
            self.fsq = FSQ(
                levels=config.fsq.levels,
                dim=self.repr_dim,
                num_codebooks=config.fsq.num_codebooks,
                project_in=config.fsq.project_in,
                eps=config.fsq.eps,
            )
        else:
            self.fsq = None

        # 4. Re-embedding層
        if config.use_fsq:
            self.fsq_embedding = FSQEmbedding(
                levels=config.fsq.levels,
                embed_dim=config.reembed_dim,
            )
            predictor_input_dim = config.reembed_dim
        else:
            self.fsq_embedding = None
            predictor_input_dim = self.repr_dim

        # 5. Discrete Predictor
        self.predictor = build_discrete_predictor(
            config=config.predictor,
            repr_dim=predictor_input_dim,
            action_dim=config.action_dim,
            num_codebooks=config.fsq.num_codebooks,
            num_levels=config.fsq.levels,
        )

    def subsampling_ratio(self):
        """HJEPAとの互換性のため"""
        return 1

    def encode(
        self,
        input_states: torch.Tensor,
        propio_pos: Optional[torch.Tensor] = None,
        propio_vel: Optional[torch.Tensor] = None,
    ):
        """エンコード（連続表現まで）

        Args:
            input_states: (T, B, C, H, W) - 入力画像

        Returns:
            z_continuous: (T, B, D) - 連続表現
        """
        # Backbone forward
        if hasattr(self.backbone, 'propio_dim') and self.backbone.propio_dim is not None:
            if propio_pos is not None and propio_pos.numel() > 0:
                if propio_vel is not None and propio_vel.numel() > 0:
                    propio_states = torch.cat([propio_pos, propio_vel], dim=-1)
                else:
                    propio_states = propio_pos
            else:
                propio_states = propio_vel

            backbone_output = self.backbone.forward_multiple(
                input_states, propio=propio_states
            )
        else:
            backbone_output = self.backbone.forward_multiple(input_states)

        z_continuous = backbone_output.encodings  # (T, B, D)

        return z_continuous, backbone_output

    def quantize(self, z_continuous: torch.Tensor):
        """量子化（連続→離散）

        Args:
            z_continuous: (T, B, D) - 連続表現

        Returns:
            z_quantized: (T, B, num_codebooks) - 量子化値
            z_indices: (T, B, num_codebooks) - 離散インデックス
        """
        if self.fsq is None:
            raise ValueError("FSQ is not enabled")

        # FSQ量子化
        fsq_output = self.fsq(z_continuous)

        return fsq_output.quantized, fsq_output.indices

    def embed_indices(self, z_indices: torch.Tensor):
        """インデックスをre-embed

        Args:
            z_indices: (T, B, num_codebooks) - 離散インデックス

        Returns:
            z_discrete: (T, B, embed_dim) - Re-embedded表現
        """
        if self.fsq_embedding is None:
            raise ValueError("FSQ embedding is not enabled")

        return self.fsq_embedding(z_indices)

    def forward_posterior(
        self,
        input_states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        propio_pos: Optional[torch.Tensor] = None,
        propio_vel: Optional[torch.Tensor] = None,
        encode_only: bool = False,
        **kwargs
    ):
        """訓練時のフォワードパス（posterior）

        Args:
            input_states: (T, B, C, H, W) - 入力画像系列
            actions: (T-1, B, A) - アクション系列
            encode_only: bool - エンコードのみ行うか

        Returns:
            DiscreteForwardResult
        """
        T = input_states.shape[0]

        # 1. エンコード（連続表現）
        z_continuous, backbone_output = self.encode(
            input_states, propio_pos, propio_vel
        )

        # 2. EMA Backbone（オプション）
        if self.backbone_ema is not None:
            if self.backbone.propio_dim is not None:
                if propio_pos is not None and propio_pos.numel() > 0:
                    if propio_vel is not None and propio_vel.numel() > 0:
                        propio_states = torch.cat([propio_pos, propio_vel], dim=-1)
                    else:
                        propio_states = propio_pos
                else:
                    propio_states = propio_vel
                ema_backbone_output = self.backbone_ema.forward_multiple(
                    input_states, propio=propio_states
                )
            else:
                ema_backbone_output = self.backbone_ema.forward_multiple(input_states)
        else:
            ema_backbone_output = None

        if encode_only:
            return DiscreteForwardResult(
                backbone_output=backbone_output,
                ema_backbone_output=ema_backbone_output,
                z_continuous=z_continuous,
            )

        # 3. FSQ量子化
        z_quantized, z_indices = self.quantize(z_continuous)

        # 4. Re-embedding
        z_discrete = self.embed_indices(z_indices)

        # 5. 予測（離散 → 離散）
        # Teacher forcing: 訓練時は正解のz_indicesを使用
        pred_output = self.predictor.forward_multiple(
            state_encs=z_discrete[0:1],  # 初期状態 (1, B, D)
            actions=actions,
            T=T - 1,
            fsq_embedding=self.fsq_embedding,
            teacher_forcing_ratio=self.config.predictor.teacher_forcing_ratio,
            ground_truth_indices=z_indices,  # Ground truth
        )

        return DiscreteForwardResult(
            backbone_output=backbone_output,
            ema_backbone_output=ema_backbone_output,
            z_continuous=z_continuous,
            z_quantized=z_quantized,
            z_indices=z_indices,
            z_discrete=z_discrete,
            pred_output=pred_output,
            actions=actions,
        )

    def forward_prior(
        self,
        input_states: torch.Tensor,
        actions: torch.Tensor,
        T: Optional[int] = None,
        repr_input: bool = False,
        **kwargs
    ):
        """推論時のフォワードパス（prior）

        Args:
            input_states: (B, C, H, W) または (B, D) - 初期状態
            actions: (T, B, A) - アクション系列
            T: int - 予測ステップ数
            repr_input: bool - 入力が既にエンコード済みか

        Returns:
            DiscreteForwardResult
        """
        if T is None:
            T = actions.shape[0]

        # 初期状態の取得
        if repr_input:
            # 既にエンコード済み
            z_continuous = input_states.unsqueeze(0)  # (1, B, D)
        else:
            # エンコード
            z_continuous, backbone_output = self.encode(input_states.unsqueeze(0))

        # 量子化してRe-embed
        z_quantized, z_indices = self.quantize(z_continuous)
        z_discrete = self.embed_indices(z_indices)

        # 予測（自己回帰）
        pred_output = self.predictor.forward_multiple(
            state_encs=z_discrete,  # (1, B, D)
            actions=actions,
            T=T,
            fsq_embedding=self.fsq_embedding,
            teacher_forcing_ratio=0.0,  # 推論時はteacher forcingなし
        )

        return DiscreteForwardResult(
            backbone_output=None,
            z_continuous=z_continuous,
            z_quantized=z_quantized,
            z_indices=z_indices,
            z_discrete=z_discrete,
            pred_output=pred_output,
            actions=actions,
        )

    def update_ema(self):
        """EMA Backboneの更新"""
        if self.backbone_ema is not None:
            for param, ema_param in zip(
                self.backbone.parameters(), self.backbone_ema.parameters()
            ):
                ema_param.data.mul_(self.config.momentum).add_(
                    param.data, alpha=1 - self.config.momentum
                )


class DiscreteHJEPA(nn.Module):
    """DiscreteJEPAのHJEPAラッパー

    既存のHJEPAと互換性を保つためのラッパークラス
    """

    def __init__(
        self,
        config,  # DiscreteHJEPAConfig
        input_dim,
        normalizer=None,
        use_propio_pos: bool = False,
        use_propio_vel: bool = False,
    ):
        super().__init__()

        self.config = config

        # Level 1: DiscreteJEPA
        self.level1 = DiscreteJEPA(
            config=config.level1,
            input_dim=input_dim,
            normalizer=normalizer,
            use_propio_pos=use_propio_pos,
            use_propio_vel=use_propio_vel,
        )

        # Level 2は未実装（disable_l2=Trueを想定）
        self.level2 = None

        # 設定
        self.train_l1 = config.train_l1
        self.freeze_l1 = config.freeze_l1
        self.disable_l2 = config.disable_l2

    @property
    def spatial_repr_dim(self):
        """HJEPAとの互換性のため"""
        return self.level1.spatial_repr_dim

    def forward(self, *args, **kwargs):
        """HJEPAと同じインターフェース"""
        return self.level1.forward_posterior(*args, **kwargs)

    def update_ema(self):
        """EMA更新"""
        self.level1.update_ema()
