# 実装ガイド - Persona Steering CMA-ES

## プロジェクト概要

このプロジェクトは、CMA-ES最適化とSVDベースのステアリングベクトルを用いた、ペルソナベースの言語モデルステアリングフレームワークです。

**GitHubリポジトリ**: https://github.com/darugareo/persona-steering-cmaes.git

---

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/darugareo/persona-steering-cmaes.git
cd persona-steering-cmaes
```

### 2. Python環境のセットアップ

```bash
# 仮想環境の作成（推奨）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係のインストール
pip install torch transformers cmaes numpy pandas scipy scikit-learn
pip install openai python-dotenv tqdm matplotlib seaborn jupyter
```

**必要なライブラリ:**
- PyTorch (CUDA対応推奨)
- Transformers (HuggingFace)
- CMA-ES
- OpenAI API

### 3. HuggingFace認証

Llama-3モデルを使用するため、HuggingFaceのアクセストークンが必要です。

```bash
# HuggingFace CLIでログイン
pip install huggingface_hub
huggingface-cli login

# または環境変数で設定
export HF_TOKEN=your_huggingface_token
```

**モデルアクセス申請:**
1. https://huggingface.co でアカウント作成
2. https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct でアクセス申請
3. https://huggingface.co/settings/tokens でトークン生成

### 4. OpenAI APIキーの設定

評価にGPT-4を使用するため、OpenAI APIキーが必要です。

```bash
# .envファイルの作成
cat > .env << EOF
OPENAI_API_KEY=sk-your-api-key-here
JUDGE_MODEL=gpt-4o-mini
GENERATOR_MODEL=gpt-4o
EOF
```

**APIキー取得:**
1. https://platform.openai.com でアカウント作成
2. https://platform.openai.com/api-keys でAPIキー生成
3. 支払い情報の登録（従量課金）

**概算コスト:**
- 1ペルソナ最適化: $0.50-$2.00
- 全13ペルソナ評価: $10-$30

---

## 実行手順

### ステップ1: SVDステアリングベクトルの構築

特性別ステアリングベクトル（R1-R5）を生成します。

```bash
python3 scripts/run_build_svd_vectors.py
```

**出力:** `data/steering_vectors_v2/` にベクトルファイルが作成されます。

### ステップ2: ペルソナ最適化の実行

特定のペルソナに対してCMA-ES最適化を実行します。

```bash
# 単一ペルソナの最適化
python3 scripts/run_persona_optimization.py \
    --persona_id episode-184019_A \
    --layer 20 \
    --alpha 2.0 \
    --iterations 50
```

**パラメータ:**
- `--persona_id`: ペルソナID（`personas/`ディレクトリ内）
- `--layer`: ステアリング層（推奨: 20）
- `--alpha`: ステアリング強度（推奨: 2.0）
- `--iterations`: 最適化イテレーション数（推奨: 50-100）

### ステップ3: ベースライン手法との比較

提案手法を他の手法と比較します。

```bash
python3 scripts/run_baseline_comparison.py \
    --persona_id episode-184019_A \
    --methods proposed meandiff pca prompt
```

**利用可能な手法:**
- `proposed`: SVD + CMA-ES（提案手法）
- `meandiff`: 平均差分ステアリング
- `pca`: PCAベースステアリング
- `prompt`: プロンプトベース（ステアリングなし）
- `random`: ランダムサーチ

### ステップ4: 評価レポートの生成

総合評価レポートを生成します。

```bash
python3 scripts/generate_phase1_report.py \
    --persona_id episode-184019_A
```

**出力:** 評価結果、統計、グラフを含むレポートが生成されます。

---

## 高度な実行例

### 複数ペルソナの一括最適化

```bash
# 7ペルソナの最適化
python3 scripts/run_7personas_optimization.py

# 10ペルソナの完全評価
python3 scripts/run_10personas_complete_evaluation.py
```

### アブレーション実験

```bash
# 層シフト実験
python3 scripts/run_layer_shift_ablation.py

# 特性シャッフル実験
python3 scripts/run_trait_shuffle_ablation.py
```

### ベンチマーク評価

```bash
# LaMP評価
python3 scripts/run_lamp_evaluation.py

# TruthfulQA評価
python3 scripts/run_truthfulqa_eval.py

# MMLU評価
python3 scripts/run_mmlu_eval.py
```

---

## プロジェクト構造

```
persona-steering-cmaes/
├── persona_opt/           # コア最適化・ステアリングモジュール
│   ├── cmaes_persona_optimizer.py
│   ├── svd_vector_builder.py
│   ├── internal_steering_l3.py
│   └── baselines/         # ベースライン手法
├── persona_judge/         # ペルソナプロファイルと評価
│   ├── persona_profile.py
│   └── judge_prompt_builder.py
├── scripts/               # 実行スクリプト（150+）
│   ├── run_build_svd_vectors.py
│   ├── run_persona_optimization.py
│   ├── run_baseline_comparison.py
│   └── generate_phase1_report.py
├── personas/              # ペルソナ設定（35ペルソナ）
│   ├── episode-184019_A/
│   │   ├── persona_profile.txt
│   │   ├── persona_features.json
│   │   ├── persona_samples.json
│   │   └── final_judge_prompt.txt
│   └── ...
├── data/                  # 共有データ
├── results/               # 実験結果
└── docs/                  # ドキュメント
```

---

## 利用可能なペルソナ

プロジェクトには35の事前設定ペルソナが含まれています:

- episode-184019_A
- episode-239427_A
- episode-118328_B
- episode-134226_A
- episode-137872_B
- episode-158821_B
- episode-179307_A
- ...他28ペルソナ

各ペルソナディレクトリには以下が含まれます:
- `persona_profile.txt` - 自然言語プロファイル
- `persona_features.json` - 抽出された特性
- `persona_samples.json` - 代表的なサンプル
- `final_judge_prompt.txt` - 評価用プロンプト
- `raw_conversations.json` - 元の会話データ

---

## システム要件

### ハードウェア

- **GPU**: 16GB VRAM以上（推奨: 24GB+）
- **CPU**: 8コア以上推奨
- **RAM**: 32GB以上推奨
- **ストレージ**: 50GB以上の空き容量

**注意**: CPU推論も可能ですが、非常に遅くなります。

### ソフトウェア

- Python 3.8以上
- CUDA 11.8以上（GPU使用時）
- Git

---

## トラブルシューティング

### モデルのダウンロードエラー

```bash
# HuggingFaceトークンの確認
huggingface-cli whoami

# モデルアクセスの確認
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')"
```

### CUDA Out of Memory

- バッチサイズを減らす
- より小さいモデルを使用
- グラディエントチェックポインティングを有効化

### OpenAI API エラー

```bash
# APIキーの確認
python3 test_openai_api.py

# レート制限に注意
# 必要に応じてスリープ時間を調整
```

---

## 出力ファイル

実験結果は以下のように整理されます:

```
results/
├── same_model/           # 同一モデル評価
├── cross_model/          # クロスモデル評価
├── judge_evaluation/     # ジャッジ評価ログ
└── lamp7/               # LaMP-7結果
```

---

## 引用

このコードを使用する場合は、以下を引用してください:

```bibtex
@article{persona_steering_cmaes,
  title={Persona-based Language Model Steering via CMA-ES Optimization},
  author={Taisei Nakata},
  year={2025}
}
```

---

## ライセンス

MITライセンス - 詳細は[LICENSE](LICENSE)ファイルを参照

Copyright (c) 2025 Taisei Nakata

---

## サポート

- **GitHub Issues**: https://github.com/darugareo/persona-steering-cmaes/issues
- **詳細ドキュメント**: `docs/`ディレクトリを参照
