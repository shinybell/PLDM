# PLDM: Planning with Latent Dynamics Models - Architecture Explanation

## Overview

PLDM (Planning with Latent Dynamics Models) is a model-based reinforcement learning approach that learns environment dynamics in a latent space from reward-free offline trajectories. The model is based on the Joint Embedding Predictive Architecture (JEPA) and uses Model Predictive Control (MPC) with MPPI optimization for planning.

**Paper**: [Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models](https://arxiv.org/abs/2502.14819)

---

## 1. Model Architecture

### 1.1 Overall Structure: HJEPA (Hierarchical JEPA)

The top-level model is `HJEPA` (defined in [pldm/models/hjepa.py:24-92](pldm/models/hjepa.py#L24-L92)), which wraps a single-level JEPA model called `level1`. The hierarchical structure was designed to support multi-level temporal abstraction, but in the current configuration (`disable_l2: true`), only `level1` is active.

```
HJEPA
└── level1 (JEPA)
    ├── backbone (Encoder)
    ├── backbone_ema (EMA Encoder, optional)
    └── predictor (RNN-based dynamics model)
```

### 1.2 JEPA (Joint-Embedding Predictive Architecture)

The core model is `JEPA` (defined in [pldm/models/jepa.py:35-245](pldm/models/jepa.py#L35-L245)), which consists of:

1. **Backbone (Encoder)**: Encodes raw observations into latent representations
2. **Backbone EMA** (optional): Exponential Moving Average version of the encoder for stability
3. **Predictor**: Predicts future latent states given current latent state and actions

#### Configuration Example (from seqlen90_3M.yaml)

```yaml
hjepa:
  level1:
    backbone:
      arch: impala                    # Encoder architecture
      backbone_subclass: i            # IMPALA variant
      backbone_width_factor: 2        # Width multiplier
      channels: 2                     # Input channels
      final_ln: true                  # Final LayerNorm
    predictor:
      predictor_arch: rnnV2           # RNN-based predictor
      predictor_subclass: '512-512'   # Hidden dimensions
      rnn_layers: 1                   # Number of RNN layers
      residual: true                  # Residual connections
      predictor_ln: true              # LayerNorm in predictor
    action_dim: 2                     # Action space dimension
    momentum: 0                       # EMA momentum (0 = disabled)
```

---

## 2. Component Details

### 2.1 Backbone (Encoder)

The encoder transforms raw observations into latent representations.

**Architecture**: IMPALA CNN ([pldm/models/encoders/impala.py](pldm/models/encoders/impala.py))

The IMPALA encoder consists of:
- Multiple residual blocks with convolutional layers
- Group normalization
- Dimensionality reduction via pooling or strided convolutions
- Optional final LayerNorm

**Input/Output**:
- Input: `(T, B, C, H, W)` where T=time, B=batch, C=channels, H/W=height/width
- Output: `(T, B, D)` where D=representation dimension (typically 512 or 1024)

**From config**:
- `channels: 2` - Input has 2 channels (grayscale + additional info)
- `img_size: 65` - Input images are 65×65 pixels
- `backbone_width_factor: 2` - Doubles the width of convolutional layers

### 2.2 Predictor (Dynamics Model)

The predictor is an RNN-based model that predicts future latent states.

**Architecture**: RNNPredictorV2 ([pldm/models/predictors.py:328-391](pldm/models/predictors.py#L328-L391))

Components:
- **GRU Cell**: Core recurrent unit for temporal dynamics
- **LayerNorm**: Applied to RNN outputs for stability
- **Residual connections**: Optional skip connections

**Forward Pass** (one step):
```python
def forward(self, rnn_state, rnn_input):
    # rnn_state: (num_layers, bs, hidden_dim)
    # rnn_input: (bs, action_dim)
    next_state, next_hidden_state = self.rnn(rnn_input.unsqueeze(0), rnn_state)
    next_state = self.final_ln(next_state)
    next_hidden_state = self.final_ln(next_hidden_state)
    return next_state[0], next_hidden_state
```

**Multi-step rollout** (forward_multiple):
- Takes initial latent state: `z_0` (from encoder)
- Rolls out for T steps using actions: `a_0, a_1, ..., a_{T-1}`
- Produces predicted latents: `ẑ_1, ẑ_2, ..., ẑ_T`

**From config**:
- `predictor_arch: rnnV2` - GRU-based predictor
- `rnn_layers: 1` - Single-layer GRU
- `predictor_subclass: '512-512'` - Hidden dimension 512
- `residual: true` - Enables residual connections

---

## 3. Tensor Processing Flow

### 3.1 Training (Forward Posterior)

During training, the model processes ground-truth observation sequences.

**Input**:
- `states`: (T, B, C, H, W) - Raw observations
- `actions`: (T-1, B, A) - Actions between states

**Processing Pipeline**:

1. **Encode all states** → Latent representations
   ```
   states (T, B, 2, 65, 65)
      ↓ backbone.forward_multiple()
   encodings (T, B, D)
   ```
   - Processes entire sequence through encoder
   - Each timestep encoded independently
   - Output: `z_0, z_1, ..., z_T`

2. **Optional: Encode with EMA backbone** (if `momentum > 0`)
   ```
   states (T, B, 2, 65, 65)
      ↓ backbone_ema.forward_multiple()
   ema_encodings (T, B, D)
   ```

3. **Predict future states** → Predicted latent sequence
   ```
   z_0 (B, D), actions (T-1, B, A)
      ↓ predictor.forward_multiple()
   predictions (T, B, D)
   ```
   - Starts from initial state `z_0`
   - Rolls out predictor using actions
   - Output: `ẑ_0, ẑ_1, ..., ẑ_T` (where ẑ_0 = z_0)

**Code location**: [pldm/models/jepa.py:161-236](pldm/models/jepa.py#L161-L236)

### 3.2 Planning (Forward Prior)

During planning/evaluation, the model rolls out from a single encoded observation.

**Input**:
- `input_states`: (B, C, H, W) or (B, D) - Single observation or latent
- `actions`: (T, B, A) - Planned action sequence

**Processing Pipeline**:

1. **Encode initial state** (if not already encoded)
   ```
   input_states (B, 2, 65, 65)
      ↓ backbone.forward_multiple()
   current_state (B, D)
   ```

2. **Rollout predictions**
   ```
   current_state (B, D), actions (T, B, A)
      ↓ predictor.forward_multiple()
   predictions (T, B, D)
   ```
   - Predicts T future latent states
   - Used for planning: evaluates action sequences

**Code location**: [pldm/models/jepa.py:110-159](pldm/models/jepa.py#L110-L159)

### 3.3 Sequence Length and Subsampling

**From config**:
- `n_steps: 16` - Training sequence length
- `l1_n_steps: 16` - Level 1 sequence length (can differ from n_steps)

During training, if the offline dataset has longer sequences, HJEPA randomly samples a subsequence:

```python
# In HJEPA.forward_posterior (hjepa.py:59-89)
sub_idx = random.randint(0, input_states.shape[0] - self.config.l1_n_steps)
l1_input_states = input_states[sub_idx : sub_idx + self.config.l1_n_steps]
l1_actions = actions[sub_idx : sub_idx + self.config.l1_n_steps - 1]
```

This means:
- Full trajectory might be 90 steps (from dataset)
- Model trains on random 16-step subsequences
- Helps prevent overfitting to specific trajectory segments

---

## 4. Loss Functions and Hyperparameters

The model is trained using two main objectives: **VICReg** and **IDM**.

### 4.1 VICReg (Variance-Invariance-Covariance Regularization)

VICReg prevents representation collapse through three components.

**Implementation**: [pldm/objectives/vicreg.py:54-199](pldm/objectives/vicreg.py#L54-L199)

#### Components:

**1. Similarity Loss** (MSE between predictions and targets)
```python
sim_loss = (ema_encodings[1:] - state_predictions[1:]).pow(2).mean()
```
- Minimizes distance between predicted latents and encoded ground-truth latents
- Targets can be from EMA encoder (if enabled) or regular encoder
- Only computed for timesteps 1 to T (not initial state)

**2. Variance Loss** (encourages feature variation)
```python
std = sqrt(x.var(dim=1) + 0.0001)  # std across batch
std_loss = mean(relu(std_margin - std))
```
- Computed on initial state encodings `z_0`
- Prevents collapse to constant representations
- Penalizes when std < `std_margin`

**3. Covariance Loss** (reduces feature redundancy)
```python
cov = einsum("bki,bkj->bij", x, x) / (batch_size - 1)
cov_loss = (cov.pow(2).sum() - diagonals) / num_features
```
- Computed on initial state encodings `z_0`
- Encourages decorrelation between feature dimensions
- Off-diagonal elements should be small

**Optional: Temporal regularization**
- `sim_loss_t`: Smoothness between consecutive encodings
- `std_loss_t`: Variance across time dimension
- `cov_loss_t`: Covariance across time dimension

**Total VICReg Loss**:
```python
total_loss = (
    sim_coeff * sim_loss +
    std_coeff * std_loss +
    cov_coeff * cov_loss +
    sim_coeff_t * sim_loss_t +
    std_coeff_t * std_loss_t +
    cov_coeff_t * cov_loss_t
)
```

#### Hyperparameters (from seqlen90_3M.yaml):

```yaml
objectives_l1:
  vicreg:
    sim_coeff: 1.0        # Weight for prediction similarity
    std_coeff: 3.9843     # Weight for variance regularization
    cov_coeff: 6.9238     # Weight for covariance regularization
    std_coeff_t: 0.24535  # Weight for temporal variance
    cov_coeff_t: 0.0      # Weight for temporal covariance (disabled)
    sim_coeff_t: 0.74242  # Weight for temporal smoothness
    std_margin: 1.0       # Minimum standard deviation
    std_margin_t: 1.0     # Minimum temporal std
    adjust_cov: true      # Normalize covariance by (num_features - 1)
```

### 4.2 IDM (Inverse Dynamics Model)

IDM trains an auxiliary network to predict actions from consecutive latent states.

**Implementation**: [pldm/objectives/idm.py:56-116](pldm/objectives/idm.py#L56-L116)

**Purpose**: Ensures latent representations encode action-relevant information.

**Architecture**: MLP that takes concatenated state pairs
```
Input: [z_t, z_{t+1}]  (concatenated)
   ↓ MLP (2*D → hidden → action_dim)
Output: predicted_action
```

**Loss**:
```python
action_loss = MSE(predicted_actions, ground_truth_actions)
total_loss = coeff * action_loss
```

**Processing**:
1. Extract consecutive encodings: `z_0, z_1, ..., z_{T-1}` and `z_1, z_2, ..., z_T`
2. Concatenate pairs: `[z_t || z_{t+1}]`
3. Predict actions: `â_t = MLP([z_t || z_{t+1}])`
4. Compare with ground-truth actions: `a_t`

#### Hyperparameters (from seqlen90_3M.yaml):

```yaml
objectives_l1:
  idm:
    coeff: 1.072          # Overall weight for IDM loss
    action_dim: 2         # Action space dimensionality
    arch: '512'           # MLP architecture (hidden dim 512)
    arch_subclass: a      # Architecture variant
    use_pred: false       # Use predicted states (false = use encoded states)
```

### 4.3 Combined Training Loss

The final training loss combines both objectives:

```python
# From train.py:375-383
loss_infos = [
    objective(batch, [forward_result.level1])
    for objective in self.objectives_l1
]
total_loss = sum([loss_info.total_loss for loss_info in loss_infos])
```

Where:
- `objectives_l1 = [VICReg, IDM]` (from config: `objectives: [VICReg, IDM]`)
- Each returns weighted loss
- Gradients backpropagate through entire model

**Typical loss magnitudes** (from training):
- VICReg sim_loss: ~0.01 - 0.1
- VICReg std_loss: ~0.0 - 0.5
- VICReg cov_loss: ~0.1 - 1.0
- IDM action_loss: ~0.001 - 0.01

---

## 5. Training Procedure

### 5.1 Data Loading

**Dataset**: Offline trajectories stored in `.npz` format

**Configuration** (from seqlen90_3M.yaml):
```yaml
data:
  dataset_type: DatasetType.Wall
  offline_wall_config:
    offline_data_path: "/pldm_envs/wall/presaved_datasets/wall-visual-config_rand_expert_40-v0.npz"
    n_steps: 16            # Sequence length
    batch_size: 64         # Batch size
    img_size: 65           # Image resolution
    lazy_load: false       # Load all data to memory
```

**Batch format**:
- `states`: (B, T, C, H, W) - Image observations
- `actions`: (B, T-1, A) - Actions
- Optional: proprioceptive state, velocity, etc.

### 5.2 Training Loop

**Location**: [pldm/train.py:313-442](pldm/train.py#L313-L442)

**Outer loop** (epochs):
```yaml
epochs: 2  # From config (normally much higher, e.g., 100)
```

**Inner loop** (batches):
For each batch in dataset:

1. **Load batch and move to GPU**
   ```python
   s = batch.states.cuda().transpose(0, 1)  # (B,T,C,H,W) → (T,B,C,H,W)
   a = batch.actions.cuda().transpose(0, 1) # (B,T-1,A) → (T-1,B,A)
   ```

2. **Forward pass**
   ```python
   forward_result = self.model.forward_posterior(s, a, **optional_fields)
   ```
   - Encodes all states → `encodings`
   - Predicts latent sequence → `predictions`
   - Returns `ForwardResult` containing both

3. **Compute losses**
   ```python
   loss_infos = [
       vicreg_objective(batch, [forward_result.level1]),
       idm_objective(batch, [forward_result.level1])
   ]
   total_loss = sum([info.total_loss for info in loss_infos])
   ```

4. **Backward pass and optimization**
   ```python
   self.optimizer.zero_grad()
   total_loss.backward()
   self.optimizer.step()
   self.model.update_ema()  # Update EMA encoder if enabled
   ```

5. **Logging** (every 100 steps)
   - Loss values
   - Learning rate
   - Training/data loading times

**No environment interaction**: The model never interacts with the environment during training—it only learns from the offline dataset.

### 5.3 Training Hyperparameters

**From seqlen90_3M.yaml**:

```yaml
# Optimization
base_lr: 0.0007               # Learning rate
optimizer_type: Adam          # Optimizer
optimizer_schedule: Cosine    # LR schedule (default, not in this config)
epochs: 2                     # Training epochs (test setting, normally 100)

# Batch settings
data.offline_wall_config.batch_size: 64
n_steps: 16                   # Sequence length per batch

# Checkpoint & evaluation
save_every_n_epochs: 5        # Checkpoint frequency (default)
eval_every_n_epochs: 20       # Evaluation frequency (default)
eval_during_training: false   # Don't evaluate during training
```

**Training steps per epoch**:
- Dataset size: 20,000 trajectories (from `data.wall_config.size: 20000`)
- Batch size: 64
- Steps per epoch: 20,000 / 64 ≈ 312 steps

**Total training**:
- If `epochs: 100`, total steps ≈ 31,200
- Each step sees 64 trajectories × 16 timesteps = 1,024 state-action pairs

### 5.4 Model Updates

**Parameter updates**:
- Standard SGD/Adam step on `backbone` and `predictor` parameters
- Gradients flow from both VICReg and IDM losses

**EMA updates** (if `momentum > 0`):
```python
# In JEPA.update_ema() (jepa.py:238-245)
for param, ema_param in zip(backbone.parameters(), backbone_ema.parameters()):
    ema_param.data = momentum * ema_param.data + (1 - momentum) * param.data
```
- In current config: `momentum: 0` → EMA disabled

### 5.5 No Online Data Collection

Key characteristic of PLDM:
- **No environment steps during training**
- **No reward signals used**
- **Purely offline learning** from pre-collected trajectories

The dataset is collected once (via expert or random policies) and stored. Training only reads from this static dataset.

---

## 6. Planning (Inference)

### 6.1 Planning Algorithm

At test time, PLDM uses **Model Predictive Control (MPC)** with **MPPI** (Model Predictive Path Integral) optimization.

**Configuration** (from seqlen90_3M.yaml):
```yaml
eval_cfg:
  wall_planning:
    level1:
      planner_type: PlannerType.MPPI
      max_plan_length: 96              # Planning horizon
      mppi:
        noise_sigma: 12                # Action noise std
        num_samples: 2000              # Number of candidate trajectories
        lambda_: 0.005                 # Temperature parameter
        z_reg_coeff: 0                 # Latent regularization (disabled)
```

### 6.2 Planning Process

At each environment step:

1. **Encode current observation**
   ```
   obs (1, C, H, W) → backbone → z_t (1, D)
   ```

2. **Sample action sequences**
   - Generate `num_samples=2000` candidate action sequences
   - Each sequence has length `max_plan_length=96`
   - Actions sampled from Gaussian noise + optional mean trajectory

3. **Rollout model for each candidate**
   ```
   For each action sequence A_i:
       z_t → predictor(z_t, A_i) → predicted trajectory [ẑ_{t+1}, ..., ẑ_{t+H}]
   ```

4. **Evaluate cost for each trajectory**
   ```
   cost = goal_cost + uncertainty_cost
   ```
   - **Goal cost**: Distance to target in latent space
   - **Uncertainty cost**: Variance across ensemble predictions (if K > 1)

5. **Weight trajectories by cost**
   ```
   weights = exp(-cost / lambda_)
   ```

6. **Compute optimal action**
   ```
   a_t = weighted_average(candidate_actions, weights)
   ```

7. **Execute first action and replan**
   - Execute `a_t` in environment
   - Observe new state
   - Repeat from step 1

**Replanning**: By default, replans at every step. Can reduce frequency for efficiency.

---

## 7. Key Design Choices

### 7.1 Why JEPA (not reconstruction)?

- **Latent prediction** focuses on control-relevant features
- **Reconstruction** wastes capacity on pixel-level details
- Empirical results show JEPA outperforms reconstruction-based models (e.g., DreamerV3)

### 7.2 Why VICReg?

- Prevents **representation collapse** (all latents become identical)
- No contrastive pairs needed (unlike SimCLR)
- Balances three objectives: similarity, variance, covariance

### 7.3 Why IDM?

- Ensures latent representations encode **action-relevant information**
- Complements VICReg by enforcing action predictability
- Common auxiliary task in self-supervised RL

### 7.4 Why GRU predictor?

- Recurrent structure naturally captures temporal dynamics
- More parameter-efficient than Transformers for short horizons
- LayerNorm + residual connections improve stability

### 7.5 Reward-free learning

- Enables learning from **unlabeled** demonstrations
- More flexible: same model can be used for multiple tasks
- Planning cost function defined at test time (not training time)

---

## 8. Summary Table

| Component | Architecture | Input Shape | Output Shape | Parameters (approx) |
|-----------|-------------|-------------|--------------|---------------------|
| **Backbone** | IMPALA CNN | (T, B, 2, 65, 65) | (T, B, D=512) | ~1M |
| **Predictor** | GRU + LN | (B, D=512), (T-1, B, A=2) | (T, B, D=512) | ~1M |
| **IDM MLP** | MLP | (B, 2*D=1024) | (B, A=2) | ~0.5M |
| **Total** | - | - | - | ~2.5M |

| Loss | Components | Hyperparameters | Purpose |
|------|-----------|-----------------|---------|
| **VICReg** | sim, std, cov | sim=1.0, std=3.98, cov=6.92 | Prediction + collapse prevention |
| **IDM** | action MSE | coeff=1.072 | Action-relevant representations |

| Training | Value | Description |
|----------|-------|-------------|
| **Dataset** | 20k trajectories | Offline, reward-free |
| **Batch size** | 64 | Trajectories per batch |
| **Sequence length** | 16 | Timesteps per trajectory |
| **Epochs** | 2 (test) / 100 (real) | Full passes through dataset |
| **Learning rate** | 0.0007 | Adam optimizer |
| **Steps/epoch** | ~312 | 20000 / 64 |

| Planning | Value | Description |
|----------|-------|-------------|
| **Algorithm** | MPPI | Sampling-based MPC |
| **Samples** | 2000 | Candidate trajectories |
| **Horizon** | 96 | Planning steps ahead |
| **Replanning** | Every step | Replan after each action |

---

## References

- **Paper**: [Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models](https://arxiv.org/abs/2502.14819)
- **Code**: [github.com/vladisai/PLDM](https://github.com/vladisai/PLDM)
- **Config**: [seqlen90_3M.yaml](pldm/configs/wall/icml/seqlen90_3M.yaml)
