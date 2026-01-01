"""MiniGrid MPC Evaluator"""
import torch
import time
import numpy as np

from pldm_envs.utils.normalizer import Normalizer
from pldm_envs.minigrid.data.minigrid_dataset import MiniGridDatasetConfig
from pldm_envs.minigrid.evaluation.create_envs import construct_eval_envs

from pldm.utils import format_seconds
from .utils import determine_terminations, calculate_success_rate
from pldm.models.jepa import JEPA
from pldm.planning.plotting import log_planning_plots, log_l1_planning_loss
from pldm.planning.mpc import MPCEvaluator
from pldm.planning.enums import PooledMPCResult
from pldm.planning.minigrid.enums import MiniGridMPCConfig, MPCReport


class MiniGridMPCEvaluator(MPCEvaluator):
    """
    MiniGrid環境でのMPC評価器

    Wall環境のMPCEvaluatorを参考に実装。
    MiniGridの長期計画タスク（ゴール到達）を評価する。
    """

    def __init__(
        self,
        config: MiniGridMPCConfig,
        jepa: JEPA,
        prober: torch.nn.Module,
        normalizer: Normalizer,
        minigrid_config: MiniGridDatasetConfig,
        quick_debug: bool = False,
        prefix: str = "",
    ):
        """
        Args:
            config: MiniGrid MPC設定
            jepa: 学習済みJEPAモデル
            prober: 学習済みProberモデル（位置推定用）
            normalizer: 正規化器
            minigrid_config: MiniGridデータセット設定
            quick_debug: デバッグモード
            prefix: ログのプレフィックス
        """
        super().__init__(
            config=config,
            model=jepa,
            prober=prober,
            normalizer=normalizer,
            quick_debug=quick_debug,
            prefix=prefix,
        )

        self.minigrid_config = minigrid_config

        # 評価用環境を構築
        self.envs = construct_eval_envs(
            seed=config.seed,
            minigrid_config=self.minigrid_config,
            n_envs=config.n_envs,
            level=config.level,
            normalizer=self.normalizer,
        )

    def evaluate(self):
        """
        MPC評価を実行

        Returns:
            mpc_data: MPCの実行結果
            report: 評価レポート
        """
        start_time = time.time()

        # MPCをチャンクで実行
        mpc_data = self._perform_mpc_in_chunks()

        elapsed_time = int(time.time() - start_time)
        print(f"mpc planning took {format_seconds(elapsed_time)}")

        # レポートを構築
        report = self._construct_report(mpc_data, elapsed_time=elapsed_time)

        # L1 planning lossをログ
        log_l1_planning_loss(result=mpc_data, prefix=self.prefix)

        # プランニングのプロットを生成（設定されている場合）
        if self.config.visualize_planning:
            log_planning_plots(
                result=mpc_data,
                report=report,
                idxs=(
                    list(range(report.errors.shape[0]))
                    if not self.quick_debug
                    else [0, 1]
                ),
                prefix=self.prefix,
                n_steps=self.config.n_steps,
                xy_action=False,  # MiniGridは離散アクション
            )

        return mpc_data, report

    def _construct_report(self, data: PooledMPCResult, elapsed_time: float = 0):
        """
        MPC結果から評価レポートを構築

        Args:
            data: MPC実行結果
            elapsed_time: 経過時間（秒）

        Returns:
            MPCReport: 評価レポート
        """
        config = self.config
        locations = data.locations
        targets = data.targets

        # 各エピソードの終了時刻を判定
        terminations = determine_terminations(
            locations, targets, config.error_threshold
        )

        # 最終誤差を計算
        final_errors = torch.stack(
            [
                (
                    locations[min(terminations[i], len(locations) - 1)][i].cpu()
                    - target.cpu()
                )
                .pow(2)
                .sum()
                .sqrt()
                for i, target in enumerate(targets)
            ]
        )

        # 成功率を計算（最終位置がゴールに近いか）
        final_locations = locations[-1]
        success_rate = calculate_success_rate(
            final_locations, targets, config.error_threshold
        )

        # 平均ステップ数を計算
        avg_steps = np.mean(terminations)

        report = MPCReport(
            error_mean=final_errors.mean(),
            errors=final_errors,
            terminations=terminations,
            planning_time=elapsed_time,
            success_rate=success_rate,
            avg_steps=avg_steps,
        )

        return report
