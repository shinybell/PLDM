"""DiscreteJEPA訓練スクリプト

FSQベースの離散JEPAを訓練するためのスクリプト
"""

import multiprocessing
import warnings

warnings.filterwarnings("ignore", message="Ill-formed record")

import os
import shutil
import dataclasses
import random
from tqdm.auto import tqdm

import torch
import numpy as np

try:
    multiprocessing.set_start_method("fork")
except:
    pass

from pldm.configs_discrete import DiscreteTrainConfig
from pldm.data.dataset_factory import DatasetFactory
from pldm.models.discrete_jepa import DiscreteHJEPA
from pldm.objectives.discrete_objectives import build_discrete_objectives
from pldm.optimizers.optimizer_factory import OptimizerFactory
from pldm.optimizers.schedulers import Scheduler
from pldm.logger import Logger, MetricTracker
from pldm.evaluation.evaluator import Evaluator
from pldm.models.quantizers import compute_codebook_usage


def seed_everything(seed):
    """全乱数シードを設定"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class DiscreteTrainer:
    """DiscreteJEPA訓練クラス"""

    def __init__(self, config: DiscreteTrainConfig):
        self.config = config

        # ロガー初期化
        Logger.run().initialize(
            output_path=self.config.output_path,
            wandb_enabled=self.config.wandb,
            project=config.run_project,
            name=config.run_name,
            group=config.run_group,
            config=dataclasses.asdict(config),
        )

        # シード設定
        seed_everything(config.seed)

        self.sample_step = 0
        self.epoch = 0
        self.step = 0

        # デバイス設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # データセット作成
        print("Creating datasets...")
        datasets = DatasetFactory(
            config.data,
            probing_cfg=config.eval_cfg.probing,
            disable_l2=config.discrete_hjepa.disable_l2,
        ).create_datasets()

        self.datasets = datasets
        self.ds = datasets.ds
        self.val_ds = datasets.val_ds

        # 入力次元の推定
        sample_data = next(iter(self.ds))
        input_dim = sample_data.states.shape[2:]
        print("Inferred input_dim:", input_dim)
        if len(input_dim) == 1:
            input_dim = input_dim[0]

        # 固有受容状態のチェック
        use_propio_pos = (
            hasattr(sample_data, "propio_pos")
            and sample_data.propio_pos is not None
            and bool(sample_data.propio_pos.shape[-1])
        )
        use_propio_vel = (
            hasattr(sample_data, "propio_vel")
            and sample_data.propio_vel is not None
            and bool(sample_data.propio_vel.shape[-1])
        )

        # モデル作成
        print("Building DiscreteJEPA model...")
        self.model = DiscreteHJEPA(
            config.discrete_hjepa,
            input_dim=input_dim,
            normalizer=self.ds.normalizer,
            use_propio_pos=use_propio_pos,
            use_propio_vel=use_propio_vel,
        ).to(self.device)

        # パラメータ数表示
        self.n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Number of parameters: {self.n_parameters:,}")

        # 損失関数
        print("Building objectives...")
        self.objectives = build_discrete_objectives(
            config=config.objectives_discrete,
            repr_dim=self.model.level1.repr_dim,
            fsq_levels=config.discrete_hjepa.level1.fsq.levels,
        ).to(self.device)

        # オプティマイザ
        optimizer_factory = OptimizerFactory(
            model=self.model,
            optimizer_type=config.optimizer_type,
            base_lr=config.base_lr,
        )
        self.optimizer = optimizer_factory.create_optimizer()

        # スケジューラ
        self.scheduler = Scheduler(
            schedule=config.optimizer_schedule,
            base_lr=config.base_lr,
            data_loader=self.ds,
            epochs=config.epochs,
            optimizer=self.optimizer,
        )

        # チェックポイント読み込み
        self.maybe_load_model()

        # freeze設定
        if self.config.discrete_hjepa.freeze_l1:
            print("Freezing level1 weights")
            for m in self.model.level1.modules():
                for p in m.parameters():
                    p.requires_grad = False

        # モデルコンパイル
        if config.compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile...")
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"Failed to compile model: {e}")

        # 評価器
        if not config.train_only:
            print("Building evaluator...")
            self.evaluator = Evaluator(
                config=config.eval_cfg,
                model=self.model,
                quick_debug=config.quick_debug,
                normalizer=self.ds.normalizer,
                epoch=0,
                probing_datasets=datasets.probing_datasets if hasattr(datasets, 'probing_datasets') else None,
                l2_probing_datasets=datasets.l2_probing_datasets if hasattr(datasets, 'l2_probing_datasets') else None,
                load_checkpoint_path=config.load_checkpoint_path or "",
                output_path=config.output_path or "",
                data_config=config.data,
            )
        else:
            self.evaluator = None

    def maybe_load_model(self):
        """チェックポイントからモデルを読み込み"""
        load_path = self.config.load_checkpoint_path

        # resume_if_possibleの場合、latest.ckptを探す
        if self.config.resume_if_possible and load_path is None:
            latest_path = os.path.join(self.config.output_path, "latest.ckpt")
            if os.path.exists(latest_path):
                load_path = latest_path
                print(f"Found checkpoint to resume: {load_path}")

        if load_path is not None and os.path.exists(load_path):
            print(f"Loading checkpoint from {load_path}")
            checkpoint = torch.load(load_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint and not self.config.eval_only:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch'] + 1
                print(f"Resuming from epoch {self.epoch}")

            return True
        return False

    def save_checkpoint(self, epoch: int, is_latest: bool = False):
        """チェックポイントを保存"""
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
        }

        if is_latest:
            path = os.path.join(self.config.output_path, "latest.ckpt")
        else:
            path = os.path.join(self.config.output_path, f"epoch_{epoch+1}.ckpt")

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def train_epoch(self, epoch: int):
        """1エポックの訓練"""
        self.model.train()
        metric_tracker = MetricTracker(window_size=100)

        pbar = tqdm(self.ds, desc=f"Epoch {epoch+1}/{self.config.epochs}")

        for batch_idx, batch in enumerate(pbar):
            # データをデバイスに移動
            observations = batch.states.to(self.device)  # (T, B, C, H, W)
            actions = batch.actions.to(self.device)  # (T-1, B, A)

            # デバッグ: 最初のバッチで形状確認
            if batch_idx == 0:
                print(f"[DEBUG] observations.shape: {observations.shape}")
                print(f"[DEBUG] actions.shape (before): {actions.shape}")

            # MiniGridデータセットはactions.shape = (B, T-1, A)で返すので、
            # (T-1, B, A)に転置する必要がある
            if actions.ndim == 3 and actions.shape[0] != observations.shape[0] - 1:
                actions = actions.transpose(0, 1)  # (B, T-1, A) -> (T-1, B, A)

            if batch_idx == 0:
                print(f"[DEBUG] actions.shape (after): {actions.shape}")

            # Propio（オプション）
            propio_pos = None
            propio_vel = None
            if hasattr(batch, 'propio_pos') and batch.propio_pos is not None:
                propio_pos = batch.propio_pos.to(self.device)
            if hasattr(batch, 'propio_vel') and batch.propio_vel is not None:
                propio_vel = batch.propio_vel.to(self.device)

            # フォワードパス
            forward_result = self.model.forward(
                input_states=observations,
                actions=actions,
                propio_pos=propio_pos,
                propio_vel=propio_vel,
            )

            # 損失計算
            losses = self.objectives(forward_result)
            total_loss = losses['total_loss']

            # バックプロパゲーション
            self.optimizer.zero_grad()
            total_loss.backward()

            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 学習率の更新
            lr = self.scheduler.adjust_learning_rate(self.step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # EMA更新
            if self.config.discrete_hjepa.level1.momentum > 0:
                self.model.update_ema()

            # メトリクス記録
            metrics = {
                'train/loss': total_loss.item(),
                'train/vicreg_loss': losses.get('vicreg_loss', torch.tensor(0.0)).item(),
                'train/discrete_loss': losses.get('discrete_prediction_loss', torch.tensor(0.0)).item(),
                'train/avg_accuracy': losses.get('avg_accuracy', 0.0),
                'train/lr': self.optimizer.param_groups[0]['lr'],
            }

            # コードブック利用率（定期的に）
            if batch_idx % 100 == 0 and forward_result.z_indices is not None:
                codebook_stats = compute_codebook_usage(
                    forward_result.z_indices,
                    self.config.discrete_hjepa.level1.fsq.levels,
                )
                metrics['train/codebook_usage'] = codebook_stats['avg_usage']

            metric_tracker.update(metrics)

            # プログレスバー更新
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'acc': f"{losses.get('avg_accuracy', 0.0):.3f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
            })

            self.step += 1

        # エポック終了時のメトリクス
        avg_metrics = metric_tracker.average()
        Logger.run().log(avg_metrics)

        return avg_metrics

    def train(self):
        """訓練ループ"""
        print("\n" + "="*70)
        print("Starting training...")
        print("="*70)

        # 初期評価
        if self.config.eval_at_beginning and self.evaluator is not None:
            print("\nRunning initial evaluation...")
            self.evaluator.epoch = -1
            eval_metrics = self.evaluator.evaluate()
            Logger.run().log(eval_metrics)

        # 訓練ループ
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch

            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"{'='*70}")

            # 訓練
            train_metrics = self.train_epoch(epoch)

            # チェックポイント保存
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, is_latest=False)

            # 最新チェックポイント更新
            self.save_checkpoint(epoch, is_latest=True)

            # 評価
            if (
                self.config.eval_during_training
                and (epoch + 1) % self.config.eval_every_n_epochs == 0
                and self.evaluator is not None
            ):
                print("\nRunning evaluation...")
                self.model.eval()
                self.evaluator.epoch = epoch
                with torch.no_grad():
                    eval_metrics = self.evaluator.evaluate()
                Logger.run().log(eval_metrics)
                self.model.train()

        # 最終評価
        if not self.config.train_only and self.evaluator is not None:
            print("\n" + "="*70)
            print("Running final evaluation...")
            print("="*70)
            self.model.eval()
            self.evaluator.epoch = self.config.epochs
            with torch.no_grad():
                eval_metrics = self.evaluator.evaluate()
            Logger.run().log(eval_metrics)

        print("\n" + "="*70)
        print("Training complete!")
        print("="*70)


def main():
    """メイン関数"""
    # 設定読み込み
    config = DiscreteTrainConfig.parse_from_command_line()

    # 出力ディレクトリ作成
    if config.output_path:
        os.makedirs(config.output_path, exist_ok=True)

        # テストディレクトリのクリーンアップ
        if "test" in config.output_dir:
            if os.path.exists(config.output_path):
                shutil.rmtree(config.output_path)
            os.makedirs(config.output_path, exist_ok=True)

        # 設定保存
        config.save(os.path.join(config.output_path, "config.yaml"))

    # トレーナー作成
    trainer = DiscreteTrainer(config)

    # 訓練実行
    if config.eval_only:
        # 評価のみ
        print("Running evaluation only...")
        if trainer.evaluator is not None:
            trainer.model.eval()
            trainer.evaluator.epoch = 0
            with torch.no_grad():
                eval_metrics = trainer.evaluator.evaluate()
            Logger.run().log(eval_metrics)
    else:
        # 訓練
        trainer.train()


if __name__ == "__main__":
    main()
