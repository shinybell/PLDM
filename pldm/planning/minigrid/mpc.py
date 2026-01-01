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
from pldm.planning import objectives_v2
from pldm.planning.planners.enums import PlannerType
from pldm.planning.planners.mppi_planner import MPPIPlanner
from pldm.planning.planners.sgd_planner import SGDPlanner
from pldm.planning.utils import normalize_actions


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

    def _construct_planner(self, n_envs: int):
        """
        MiniGrid用のプランナーを構築

        離散アクション用に action_normalizer を調整
        """
        config = self.config

        objective = objectives_v2.ReprTargetMPCObjective(
            model=self.model,
            propio_cost=config.level1.propio_cost,
            sum_all_diffs=config.level1.sum_all_diffs,
            loss_coeff_first=config.level1.loss_coeff_first,
            loss_coeff_last=config.level1.loss_coeff_last,
        )

        # MiniGridは離散アクション（one-hot）なので xy_action=False
        action_normalizer = lambda x: normalize_actions(
            x,
            min_norm=config.level1.min_step,
            max_norm=config.level1.max_step,
            xy_action=False,  # MiniGridは離散アクション
            clamp_actions=config.level1.clamp_actions,
        )

        if config.level1.planner_type == PlannerType.MPPI:
            planner = MPPIPlanner(
                config.level1.mppi,
                model=self.model,
                normalizer=self.normalizer,
                objective=objective,
                prober=self.prober,
                action_normalizer=action_normalizer,
                n_envs=n_envs,
                projected_cost=config.level1.projected_cost,
            )
        elif config.level1.planner_type == PlannerType.SGD:
            planner = SGDPlanner(
                config.level1.sgd,
                model=self.model,
                normalizer=self.normalizer,
                objective=objective,
                prober=self.prober,
                action_normalizer=action_normalizer,
            )
        else:
            raise NotImplementedError(
                f"Unknown planner type {config.level1.planner_type}"
            )

        return planner

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

    def _perform_mpc(self, planner, envs):
        """
        MiniGrid用のMPC実行

        MiniGridではゴール位置が環境内に固定されているため、
        環境からゴール位置を取得して目標とする。
        """
        # ゴール位置を取得（MiniGrid環境の内部状態から）
        targets = []
        for env in envs:
            # ラッパーを剥がして元のMiniGrid環境にアクセス
            unwrapped_env = env
            while hasattr(unwrapped_env, 'env'):
                unwrapped_env = unwrapped_env.env

            # ゴール位置を取得
            goal_pos = unwrapped_env.unwrapped.goal_pos
            targets.append(np.array(goal_pos, dtype=np.float32))

        targets = torch.from_numpy(np.stack(targets)).to(self.device)

        # 初期観測を取得（先にリセットして環境を初期化）
        observation_history = []
        for env in envs:
            obs, _ = env.reset()
            observation_history.append(torch.from_numpy(obs).float())
        observation_history = [torch.stack(observation_history).to(self.device)]

        # 目標観測を取得（環境がリセットされた後）
        # MiniGridでは、目標位置にエージェントを仮想的に配置してレンダリング
        # DiverseMazeの get_target_obs() と同じアプローチ
        target_obs_list = []
        for i, env in enumerate(envs):
            # 環境のget_target_obs()を使って目標位置での観測を取得
            target_obs = env.get_target_obs()  # (H, W, 3) uint8
            target_obs_list.append(torch.from_numpy(target_obs).float())

        target_obs_batch = torch.stack(target_obs_list).to(self.device)  # (bs, H, W, 3)

        # 正規化
        if self.normalizer is not None:
            target_obs_batch = self.normalizer.normalize_state(target_obs_batch)

        # バックボーンで目標観測をエンコード
        with torch.no_grad():
            target_enc = self.model.backbone(target_obs_batch).obs_component.detach()

        # エンコードされた目標表現をプランナーに設定
        planner.reset_targets(target_enc, repr_input=True)

        obs_t = observation_history[0]
        if self.image_based:
            obs_t = torch.cat([obs_t] * self.config.stack_states, dim=1)

        action_history = []
        reward_history = []
        location_history = []
        pred_location_history = []
        loss_history = []

        # エージェントの初期位置を記録
        initial_locations = []
        for env in envs:
            unwrapped_env = env
            while hasattr(unwrapped_env, 'env'):
                unwrapped_env = unwrapped_env.env
            agent_pos = unwrapped_env.unwrapped.agent_pos
            initial_locations.append(np.array(agent_pos, dtype=np.float32))
        location_history.append(torch.from_numpy(np.stack(initial_locations)).to(self.device))

        # MPCループ
        for step in range(self.config.n_steps):
            # 現在の観測をエンコード
            if self.normalizer is not None:
                obs_t_norm = self.normalizer.normalize_state(obs_t)
            else:
                obs_t_norm = obs_t

            with torch.no_grad():
                obs_encoded = self.model.backbone(obs_t_norm).obs_component.detach()

            # プランニング
            actions, info = planner.plan(obs_encoded)

            # 最初のアクションを実行
            action = actions[:, 0]  # (bs, action_dim)

            # 離散アクションに変換（one-hotから離散値へ）
            action_indices = torch.argmax(action, dim=-1).cpu().numpy()

            # 環境でアクションを実行
            next_obs_list = []
            rewards = []
            current_locations = []

            for i, env in enumerate(envs):
                obs, reward, done, truncated, info_dict = env.step(int(action_indices[i]))
                next_obs_list.append(torch.from_numpy(obs).float())
                rewards.append(reward)

                # 現在位置を取得
                unwrapped_env = env
                while hasattr(unwrapped_env, 'env'):
                    unwrapped_env = unwrapped_env.env
                agent_pos = unwrapped_env.unwrapped.agent_pos
                current_locations.append(np.array(agent_pos, dtype=np.float32))

            # 記録
            action_history.append(action.cpu())
            reward_history.append(torch.tensor(rewards))
            location_history.append(torch.from_numpy(np.stack(current_locations)).to(self.device))

            if 'pred_locations' in info:
                pred_location_history.append(info['pred_locations'])
            if 'loss_history' in info:
                loss_history.append(info['loss_history'])

            # 次の観測を準備
            obs_t = torch.stack(next_obs_list).to(self.device)
            if self.image_based:
                obs_t = torch.cat([obs_t] * self.config.stack_states, dim=1)
            observation_history.append(obs_t)

        # 結果をPooledMPCResultにまとめる
        result = PooledMPCResult(
            observations=observation_history,
            locations=location_history,
            actions=action_history,
            rewards=reward_history,
            pred_locations=pred_location_history if pred_location_history else None,
            targets=targets.cpu(),
            loss_history=loss_history if loss_history else None,
        )

        return result
