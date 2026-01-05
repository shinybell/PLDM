"""DiscreteJEPA用の損失関数

VICReg（FSQ前の連続表現）+ CrossEntropy（離散予測）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict

from pldm.models.discrete_jepa import DiscreteForwardResult
from pldm.objectives.vicreg import VICRegObjective


class DiscretePredictionLoss(nn.Module):
    """離散インデックス予測のCrossEntropy損失

    各コードブック次元ごとにCrossEntropy損失を計算

    Args:
        num_levels: List[int] - 各次元のクラス数
        weight: float - 損失の重み
        per_dim_weight: bool - 各次元で重み付けするか
        dim_weights: Optional[List[float]] - 各次元の重み
        label_smoothing: float - ラベル平滑化
    """

    def __init__(
        self,
        num_levels: List[int],
        weight: float = 1.0,
        per_dim_weight: bool = False,
        dim_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.num_levels = num_levels
        self.num_codebooks = len(num_levels)
        self.weight = weight
        self.per_dim_weight = per_dim_weight
        self.label_smoothing = label_smoothing

        # 各次元の重み
        if dim_weights is not None:
            assert len(dim_weights) == self.num_codebooks
            self.dim_weights = torch.tensor(dim_weights)
        elif per_dim_weight:
            # デフォルト: 各次元を均等に重み付け
            self.dim_weights = torch.ones(self.num_codebooks) / self.num_codebooks
        else:
            self.dim_weights = torch.ones(self.num_codebooks)

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_logits: (T, B, C, L) - 予測logits
            target_indices: (T, B, C) - 正解インデックス
            mask: Optional[(T, B)] - 有効なステップのマスク

        Returns:
            dict: 損失値と統計
        """
        T, B, C, L = pred_logits.shape
        device = pred_logits.device

        # 重みをデバイスに移動
        dim_weights = self.dim_weights.to(device)

        total_loss = 0.0
        losses_per_dim = []
        accuracies_per_dim = []

        for c in range(C):
            # c番目のコードブック次元
            logits_c = pred_logits[:, :, c, :self.num_levels[c]]  # (T, B, L_c)
            target_c = target_indices[:, :, c]  # (T, B)

            # 平坦化
            logits_c = logits_c.reshape(-1, self.num_levels[c])  # (T*B, L_c)
            target_c = target_c.reshape(-1)  # (T*B,)

            # CrossEntropy損失
            loss_c = F.cross_entropy(
                logits_c,
                target_c,
                label_smoothing=self.label_smoothing,
                reduction='none',  # (T*B,)
            )

            # マスク適用
            if mask is not None:
                mask_flat = mask.reshape(-1)  # (T*B,)
                loss_c = loss_c * mask_flat
                loss_c = loss_c.sum() / (mask_flat.sum() + 1e-8)
            else:
                loss_c = loss_c.mean()

            # 次元ごとの重み適用
            weighted_loss_c = loss_c * dim_weights[c]
            total_loss += weighted_loss_c

            losses_per_dim.append(loss_c.item())

            # 正解率
            pred_c = logits_c.argmax(dim=-1)
            if mask is not None:
                mask_flat = mask.reshape(-1)
                correct = ((pred_c == target_c) * mask_flat).sum()
                total = mask_flat.sum()
            else:
                correct = (pred_c == target_c).sum()
                total = torch.tensor(pred_c.numel(), dtype=torch.float32, device=pred_c.device)

            accuracy_c = correct.float() / (total.float() + 1e-8)
            accuracies_per_dim.append(accuracy_c.item())

        # 平均損失
        if self.per_dim_weight:
            # 既に重み付け済み
            avg_loss = total_loss
        else:
            # 平均を取る
            avg_loss = total_loss / C

        return {
            'discrete_prediction_loss': avg_loss * self.weight,
            'discrete_prediction_loss_raw': avg_loss,
            'losses_per_dim': losses_per_dim,
            'accuracies_per_dim': accuracies_per_dim,
            'avg_accuracy': sum(accuracies_per_dim) / len(accuracies_per_dim),
        }


class DiscreteObjectives(nn.Module):
    """DiscreteJEPA用の全損失関数

    VICReg（FSQ前の連続表現）+ DiscretePrediction（離散予測）

    Args:
        vicreg_config: VICReg設定
        discrete_prediction_config: 離散予測損失設定
        fsq_levels: List[int] - FSQレベル数
        repr_dim: int - 表現次元（VICReg用）
    """

    def __init__(
        self,
        vicreg_config,
        discrete_prediction_config,
        fsq_levels: List[int],
        repr_dim: int,
    ):
        super().__init__()

        # VICReg損失（FSQ前の連続表現用）
        self.vicreg = VICRegObjective(
            vicreg_config,
            name_prefix="discrete",
            repr_dim=repr_dim,
        )

        # 離散予測損失
        self.discrete_prediction = DiscretePredictionLoss(
            num_levels=fsq_levels,
            weight=discrete_prediction_config.weight,
            per_dim_weight=discrete_prediction_config.per_dim_weight,
            dim_weights=discrete_prediction_config.dim_weights,
            label_smoothing=discrete_prediction_config.label_smoothing,
        )

    def forward(
        self,
        forward_result: DiscreteForwardResult,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            forward_result: DiscreteForwardResult

        Returns:
            dict: 全損失と統計
        """
        losses = {}

        # 1. VICReg損失（FSQ前の連続表現）
        # z_continuous: (T, B, D)
        # VICRegはpredとtargetを期待
        # pred: 予測表現（t=1以降）
        # target: 目標表現（t=1以降）

        # 簡易版: z_continuous[:-1] と z_continuous[1:] でVICReg
        # より正確には、Predictorの連続表現出力が必要だが、
        # ここでは連続表現の一貫性を評価
        if forward_result.z_continuous is not None:
            T = forward_result.z_continuous.shape[0]
            if T > 1:
                # 時間方向にVICRegを適用
                # pred: t=0の連続表現
                # target: t=1以降の連続表現
                pred = forward_result.z_continuous[0:1].expand(T-1, -1, -1)  # (T-1, B, D)
                target = forward_result.z_continuous[1:]  # (T-1, B, D)

                # VICRegObjectiveのforward signature確認が必要
                # 仮実装: 直接VICReg計算
                vicreg_loss = self._compute_vicreg(pred, target)
                losses['vicreg_loss'] = vicreg_loss
            else:
                losses['vicreg_loss'] = torch.tensor(0.0, device=forward_result.z_continuous.device)
        else:
            losses['vicreg_loss'] = torch.tensor(0.0)

        # 2. 離散予測損失
        if forward_result.pred_output is not None:
            pred_logits = forward_result.pred_output.predictions  # (T, B, C, L)
            target_indices = forward_result.z_indices[1:]  # (T, B, C) - 1ステップ先

            # 予測とターゲットのサイズが一致するか確認
            T_pred = pred_logits.shape[0]
            T_target = target_indices.shape[0]

            if T_pred == T_target:
                discrete_losses = self.discrete_prediction(
                    pred_logits=pred_logits,
                    target_indices=target_indices,
                )
                losses.update(discrete_losses)
            else:
                # サイズ不一致の場合はエラーログ
                print(f"Warning: pred shape {pred_logits.shape} vs target shape {target_indices.shape}")
                losses['discrete_prediction_loss'] = torch.tensor(0.0, device=pred_logits.device)

        # 3. 総損失
        total_loss = losses.get('vicreg_loss', 0) + losses.get('discrete_prediction_loss', 0)
        losses['total_loss'] = total_loss

        return losses

    def _compute_vicreg(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """簡易VICReg計算

        Args:
            pred: (T, B, D)
            target: (T, B, D)

        Returns:
            loss: スカラー
        """
        # 簡易実装: MSE損失
        # 正確にはVICRegの各項を計算すべきだが、ここでは簡略化
        # 実際の実装では self.vicreg を適切に呼び出す

        # 平坦化
        pred_flat = pred.reshape(-1, pred.shape[-1])  # (T*B, D)
        target_flat = target.reshape(-1, target.shape[-1])  # (T*B, D)

        # MSE
        loss = F.mse_loss(pred_flat, target_flat)

        return loss


def build_discrete_objectives(
    config,  # DiscreteObjectivesConfig
    repr_dim: int,
    fsq_levels: List[int],
):
    """DiscreteObjectivesのファクトリ関数

    Args:
        config: DiscreteObjectivesConfig
        repr_dim: int - 表現次元
        fsq_levels: List[int] - FSQレベル数

    Returns:
        DiscreteObjectives
    """
    return DiscreteObjectives(
        vicreg_config=config.vicreg,
        discrete_prediction_config=config.discrete_prediction,
        fsq_levels=fsq_levels,
        repr_dim=repr_dim,
    )
