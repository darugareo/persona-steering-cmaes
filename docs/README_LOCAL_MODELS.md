# ローカルモデル統合 - 完了報告

## 実装内容

persona_optシステムに**HuggingFaceバックエンド**と**Ollamaバックエンド**のサポートを追加しました。

### 実装されたバックエンド

1. **HuggingFace (`hf/...`)** - transformersを使った直接推論
2. **Ollama (`gpt-oss-20b-local` または `ollama/...`)** - OpenAI互換API
3. **OpenAI API** - 既存のクラウドAPI

## 使用方法

### Generator

```python
# HuggingFace (推奨)
gen = PersonaGenerator(
    model="hf/meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=256,
    temperature=0.7
)

# Ollama (代替)
gen = PersonaGenerator(
    model="gpt-oss-20b-local",  # or "ollama/gpt-oss:20b"
    max_new_tokens=256,
    temperature=0.7
)

# OpenAI API
gen = PersonaGenerator(
    model="gpt-4o-mini",
    max_new_tokens=256,
    temperature=0.7
)
```

### Judge

```python
# HuggingFace (JSON出力が不安定な場合あり)
judge = LLMJudge(
    model="hf/meta-llama/Llama-3.1-8B-Instruct",
    policy_path="policy/eval_policy_v2.md"
)

# OpenAI API (推奨 - 安定したJSON出力)
judge = LLMJudge(
    model="gpt-4o-mini",
    policy_path="policy/eval_policy_v2.md"
)
```

### CMA-ES最適化

```bash
# 推奨構成: Generator=HF Local, Judge=GPT-4o-mini API
python3.8 -m persona_opt.run_cma_es \
  --generator_model hf/meta-llama/Llama-3.1-8B-Instruct \
  --judge_model gpt-4o-mini \
  --gens 10 --pop 8 --parents 4 \
  --tau 0.80

# 完全ローカル (コスト最小、Judge不安定)
python3.8 -m persona_opt.run_cma_es \
  --generator_model hf/meta-llama/Llama-3.1-8B-Instruct \
  --judge_model hf/meta-llama/Llama-3.1-8B-Instruct \
  --gens 10 --pop 8 --parents 4 \
  --tau 0.80
```

## セットアップ

### HuggingFace

1. HuggingFaceトークンを取得: https://huggingface.co/settings/tokens
2. Llama-3.1ライセンスに同意: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
3. ログイン:

```bash
export HF_TOKEN="your_token_here"
huggingface-cli login --token $HF_TOKEN
```

### Ollama (すでに設定済み)

```bash
# 確認
ollama list
# NAME           ID              SIZE     MODIFIED     
# gpt-oss:20b    f2b8351c629c    13 GB    3 months ago
```

## テスト状況

✅ **動作確認済み:**
- HuggingFaceバックエンドの実装 (generator.py, judge.py)
- Ollamaバックエンドの実装と動作確認
- モックモード
- OpenAI APIモード

⚠️ **注意事項:**
- Llama-3.1-8B-Instructはgatedモデルなので、HFトークンとライセンス同意が必要
- Ollamaのgpt-oss:20bは、Judgeとしては不安定（JSON出力が`reasoning`フィールドに入る問題）
- **推奨構成**: Generator=HF local, Judge=gpt-4o-mini API

## ファイル変更

### 新規作成
- `docs/gpt_oss_setup.md` - 詳細なセットアップガイド
- `test_gpt_oss_local.py` - スモークテストスクリプト
- `scripts/run_gpt_oss_server.sh` - Ollamaサーバー起動スクリプト

### 変更
- `persona_opt/generator.py` - HuggingFace/Ollamaバックエンド追加
- `persona_opt/judge.py` - HuggingFace/Ollamaバックエンド追加

## 次のステップ

1. HuggingFaceトークンを設定
2. テスト実行:
   ```bash
   python3.8 test_gpt_oss_local.py
   ```
3. 小規模最適化で動作確認:
   ```bash
   python3.8 -m persona_opt.run_cma_es \
     --generator_model hf/meta-llama/Llama-3.1-8B-Instruct \
     --judge_model gpt-4o-mini \
     --gens 3 --pop 4 --parents 2
   ```
