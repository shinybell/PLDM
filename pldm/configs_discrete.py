"""DiscreteJEPA用の設定ファイル

既存のconfigsと互換性を保ちながら、DiscreteJEPA固有の設定を追加
"""

from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import MISSING

from pldm.configs import ConfigBase
from pldm.data.enums import DataConfig
from pldm.models.encoders.enums import BackboneConfig
from pldm.optimizers.optimizer_factory import OptimizerType
from pldm.optimizers.schedulers import LRSchedule
from pldm.evaluation.evaluator import EvalConfig


@dataclass
class FSQConfig(ConfigBase):
    """FSQ (Finite Scalar Quantization) の設定"""
    levels: List[int] = field(default_factory=lambda: [8, 8, 8, 5, 5, 5])
    num_codebooks: int = 6  # len(levels)

    # FSQの実装詳細
    project_in: bool = True  # 入力を射影してからFSQ適用
    eps: float = 1e-5  # 数値安定性のため


@dataclass
class DiscretePredictorConfig(ConfigBase):
    """離散Predictor用の設定"""
    predictor_arch: str = "discrete_rnn"  # discrete_rnn, discrete_transformer
    predictor_subclass: str = "512-512"

    # RNN設定
    rnn_layers: int = 1
    rnn_hidden_dim: int = 512
    rnn_state_dim: int = 512  # HJEPAとの互換性のため

    # その他
    residual: bool = True
    predictor_ln: bool = True
    tie_backbone_ln: bool = True

    # 離散予測特有
    predict_per_codebook: bool = True  # 各コードブック次元を個別予測
    use_gumbel_softmax: bool = False   # Gumbel-Softmaxで微分可能にするか
    gumbel_temperature: float = 1.0

    # Teacher forcing設定
    teacher_forcing_ratio: float = 1.0  # 初期値、徐々に減衰させる
    teacher_forcing_decay: float = 0.0  # エポックごとの減衰率


@dataclass
class DiscreteJEPAConfig(ConfigBase):
    """DiscreteJEPA用の設定（JEPAConfigの代替）"""
    # FSQ設定
    fsq: FSQConfig = field(default_factory=FSQConfig)
    reembed_dim: int = 512  # Re-embedding後の次元

    # Backbone設定（既存と同じ構造を使用）
    backbone: BackboneConfig = field(default_factory=BackboneConfig)

    # Discrete Predictor設定
    predictor: DiscretePredictorConfig = field(default_factory=DiscretePredictorConfig)

    # その他（既存JEPAConfigと互換性を保つ）
    action_dim: int = 7
    momentum: float = 0.0

    # DiscreteJEPA固有
    use_fsq: bool = True  # FSQを使うかどうか（デバッグ用にfalseも可能）
    dual_output: bool = False  # 連続+離散の両方を出力するか


@dataclass
class VICRegObjectiveConfig(ConfigBase):
    """VICReg損失の設定（既存と互換性のため再定義）"""
    projector: str = "id"
    random_projector: bool = False

    # 空間次元の係数
    sim_coeff: float = 1.0
    std_coeff: float = 3.0
    cov_coeff: float = 6.9238

    # 時間次元の係数
    std_coeff_t: float = 0.24535
    cov_coeff_t: float = 0.0
    sim_coeff_t: float = 0.74242

    # その他の設定
    cov_per_feature: bool = False
    adjust_cov: bool = True
    cov_chunk_size: Optional[int] = None
    std_margin: float = 1.0
    std_margin_t: float = 1.0


@dataclass
class DiscretePredictionConfig(ConfigBase):
    """離散予測損失の設定"""
    weight: float = 1.0  # VICRegとのバランス
    per_dim_weight: bool = False  # 各コードブック次元で重み付け
    label_smoothing: float = 0.0  # ラベル平滑化

    # 各次元の重み（オプション）
    dim_weights: Optional[List[float]] = None


@dataclass
class DiscreteObjectivesConfig(ConfigBase):
    """DiscreteJEPA用の損失関数設定"""
    objectives: List[str] = field(default_factory=lambda: ["VICReg", "DiscretePrediction"])

    # VICReg（FSQ前の連続表現用）
    vicreg: VICRegObjectiveConfig = field(default_factory=VICRegObjectiveConfig)

    # 離散予測損失
    discrete_prediction: DiscretePredictionConfig = field(default_factory=DiscretePredictionConfig)


@dataclass
class DiscreteHJEPAConfig(ConfigBase):
    """DiscreteJEPA用のHJEPA相当の設定"""
    level1: DiscreteJEPAConfig = field(default_factory=DiscreteJEPAConfig)
    step_skip: int = 4
    disable_l2: bool = True  # 離散版ではL2は使わない想定
    freeze_l1: bool = False
    train_l1: bool = True
    l1_n_steps: int = 17


@dataclass
class DiscreteTrainConfig(ConfigBase):
    """DiscreteJEPA訓練用の設定"""
    # 基本設定
    env_name: str = MISSING
    n_steps: int = 17
    val_n_steps: int = 17
    l1_n_steps: int = 17

    # WandB設定
    wandb: bool = False
    run_name: Optional[str] = None
    run_group: Optional[str] = None
    run_project: str = "PLDM-Discrete"

    # 出力設定
    output_root: Optional[str] = None
    output_dir: Optional[str] = None

    # デバッグ
    quick_debug: bool = False
    seed: int = 42

    # チェックポイント
    load_checkpoint_path: Optional[str] = None
    load_l1_only: bool = False

    # 訓練/評価モード
    eval_only: bool = False
    train_only: bool = False

    # ハイパーパラメータ
    epochs: int = 100
    base_lr: float = 0.001
    disable_l2: bool = True
    optimizer_type: OptimizerType = OptimizerType.Adam
    optimizer_schedule: LRSchedule = LRSchedule.Cosine

    # データ設定（既存のDataConfigを再利用）
    data: DataConfig = field(default_factory=DataConfig)

    # モデル設定
    discrete_hjepa: DiscreteHJEPAConfig = field(default_factory=DiscreteHJEPAConfig)

    # 損失関数設定
    objectives_discrete: DiscreteObjectivesConfig = field(default_factory=DiscreteObjectivesConfig)

    # 評価設定
    eval_at_beginning: bool = False
    eval_during_training: bool = False
    save_every_n_epochs: int = 5
    eval_every_n_epochs: int = 20
    eval_cfg: EvalConfig = field(default_factory=EvalConfig)

    # その他
    resume_if_possible: bool = True
    compile_model: bool = True

    # 計算されるパス
    output_path: Optional[str] = None

    def __post_init__(self):
        """既存のTrainConfigと同様の初期化処理"""
        import os

        if self.quick_debug:
            self.data.quick_debug = True

            # エポック数を減らす
            self.epochs = min(self.epochs, 3)

            # MiniGrid用のデバッグ設定
            if hasattr(self.data, 'minigrid_config'):
                self.data.minigrid_config.quick_debug = True
                # データ数を大幅に制限（batch_size * 2 = 32サンプル程度）
                self.data.minigrid_config.crop_length = self.data.minigrid_config.batch_size * 2
                print(f"[quick_debug] Setting crop_length to {self.data.minigrid_config.crop_length}")

            # 評価設定の簡略化
            if hasattr(self.eval_cfg, 'minigrid_planning'):
                self.eval_cfg.minigrid_planning.n_envs = 2
                self.eval_cfg.minigrid_planning.n_steps = 10
                self.eval_cfg.minigrid_planning.max_episode_steps = 50

            # Probingも簡略化
            if hasattr(self.eval_cfg, 'probing'):
                self.eval_cfg.probing.epochs = 3
                self.eval_cfg.probing.sample_timesteps = 10

        # val_n_stepsを同期
        self.val_n_steps = self.n_steps

        # 評価設定
        self.eval_cfg.eval_l2 = False  # 離散版ではL2なし

        # 出力パスの設定
        if self.output_root and self.output_dir:
            self.output_path = os.path.join(
                self.output_root.rstrip("/"), self.output_dir.lstrip("/")
            )
            self.run_group = self.output_dir

        # train_onlyの場合は評価を無効化
        if self.train_only:
            self.eval_cfg.eval_l1 = False
            if hasattr(self.eval_cfg, 'probe_preds'):
                self.eval_cfg.probe_preds = False
            if hasattr(self.eval_cfg, 'probe_encoder'):
                self.eval_cfg.probe_encoder = False
            if hasattr(self.eval_cfg, 'disable_planning'):
                self.eval_cfg.disable_planning = True
