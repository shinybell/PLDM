# DiverseMaze vs MiniGrid 観測画像の比較

## DiverseMaze (maze2d-small-diverse)

### 画像処理パイプライン:
1. **元画像**: MuJoCo環境のトップビューレンダリング (約500x500)
2. **クロップ**: CenterCrop(346) → 346x346
3. **リサイズ**: Resize(64) → 64x64 RGB
4. **チャンネル順序**: (H, W, C) → (C, H, W)
5. **正規化**: Normalizer適用（オプション）

### 特徴:
- 2D平面の迷路を上から見た図
- エージェント（青い点）とゴール（赤い点）が表示
- 壁は黒、通路は白/グレー
- シンプルな幾何学的な見た目

### コード参照:
- `pldm_envs/diverse_maze/transforms.py`: Line 67-72 (small_diverse_transforms)
- `pldm_envs/diverse_maze/maze_draw.py`: Line 135-161 (render_umaze)

---

## MiniGrid

### 画像処理パイプライン:
1. **元画像**: MiniGrid環境のrender() (約448x448)
2. **クロップ**: なし
3. **リサイズ**: Resize(72) → 72x72 RGB (現在の設定)
4. **チャンネル順序**: (H, W, C) → (C, H, W)
5. **正規化**: [0, 1]に正規化（データセット側で実施）

### 特徴:
- グリッドベースの環境を上から見た図
- タイルベースの描画（壁、床、オブジェクト、エージェント）
- より構造化された見た目
- カラフルなオブジェクト（鍵、ドア、ゴールなど）

### コード参照:
- `pldm_envs/minigrid/wrappers.py`: Line 113-181 (ResizeObservationWrapper)
- `pldm_envs/minigrid/data_generation/generate_data.py`: 画像生成ロジック

---

## 主な違い

| 項目 | DiverseMaze | MiniGrid |
|------|-------------|----------|
| 最終サイズ | 64x64 | 72x72 (設定可能) |
| 環境タイプ | 連続2D迷路 | グリッドワールド |
| レンダリング | MuJoCo物理エンジン | MiniGrid独自 |
| 視覚的複雑さ | シンプル（点と線） | 中程度（タイルとオブジェクト） |
| アクション空間 | 連続（2D速度） | 離散（7アクション） |
| クロップ | あり（346x346） | なし |

## 共通点

1. **両方とも鳥瞰図（トップビュー）**: エージェントを上から見た2D画像
2. **RGB画像**: 3チャンネルカラー画像
3. **正規化**: 両方とも [0, 1] に正規化
4. **Goal-conditioned**: ゴール位置への到達を目標とする
5. **ナビゲーションタスク**: 障害物を避けてゴールに到達

## 結論

DiverseMazeとMiniGridは、どちらも**小さくリサイズされたRGB鳥瞰図**を観測として使用しています。
主な違いは環境の性質（連続 vs グリッド）とレンダリング方法ですが、
PLDMのアーキテクチャ（IMPALA CNN + RNN predictor）は両方に対応できる設計になっています。

画像サイズが64 vs 72の違いは、バックボーンCNNの入力サイズとして許容範囲内です。
