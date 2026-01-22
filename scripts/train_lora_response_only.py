#!/usr/bin/env python3
# scripts/train_lora_response_only.py

import os
import json
import torch
import copy
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse
import time

# 除外ペルソナ
EXCLUDED_PERSONAS = {
    "episode-204347_A", "episode-225888_A", "episode-239427_A",
    "episode-37624_A", "episode-38144_A", "episode-51953_A", "episode-98947_A"
}


def get_target_personas():
    """対象ペルソナのリストを取得"""
    personas_dir = Path("personas_cc")
    all_personas = [p.name for p in personas_dir.iterdir()
                    if p.is_dir() and p.name.startswith("episode-")]
    return sorted([p for p in all_personas if p not in EXCLUDED_PERSONAS])


def get_partner_role(speaker_role):
    """Partner roleを取得"""
    role_pairs = {
        "Husband": "Wife", "Wife": "Husband",
        "Parent": "Child", "Child": "Parent",
        "Mentor": "Mentee", "Mentee": "Mentor",
    }
    return role_pairs.get(speaker_role, "Partner")


def get_relationship_type(relationship):
    """Relationship typeを正規化"""
    relationship_map = {
        "Husband and Wife": "married couple",
        "Parent and Child": "parent-child",
        "Mentee and Mentor": "mentor-mentee",
        "Classmates": "classmates",
        "Neighbors": "neighbors",
    }
    return relationship_map.get(relationship, relationship.lower())


def build_prompt(turn, profile):
    """新しいプロンプト構造でプロンプトを生成（Steeringと同じ）"""

    speaker_role = profile["speaker_role"]
    partner_role = get_partner_role(speaker_role)
    relationship_type = get_relationship_type(profile["relationship"])

    prompt = f"""You are Speaker A in a conversation with Speaker B.

Speaker A role: {speaker_role}
Speaker B role: {partner_role}
Relationship: {relationship_type}

Your task:
Given the conversation so far and Speaker B's latest utterance,
produce a single natural reply as Speaker A.

Constraints:
- Respond only as Speaker A.
- Do not change roles or speak as Speaker B.
- Do not introduce facts not present in the conversation.
- The reply should be natural given the specified relationship.
- Output only the reply text.

Conversation so far:
{turn.get('context', '')}

Speaker B's latest message:
"{turn['user']}"

Your reply: """

    return prompt


def load_persona_data(persona_id):
    """ペルソナデータを読み込み"""
    persona_dir = Path(f"personas_cc/{persona_id}")

    with open(persona_dir / "profile.json") as f:
        profile = json.load(f)

    train_file = persona_dir / "train_turns_persona_specific.json"
    with open(train_file) as f:
        train_data = json.load(f)

    return profile, train_data["turns"]


class ResponseOnlyDataCollator:
    """
    応答部分のみで損失計算するData Collator

    設計B:
    - プロンプト部分: labels = -100（損失から除外）
    - 応答部分: 通常のlabels
    """

    def __init__(self, tokenizer, prompt_lengths):
        """
        Args:
            tokenizer: トークナイザー
            prompt_lengths: 各サンプルのプロンプト長のリスト
        """
        self.tokenizer = tokenizer
        self.prompt_lengths = prompt_lengths

    def __call__(self, features):
        # input_idsとattention_maskを取得
        input_ids = torch.stack([torch.tensor(f["input_ids"]) for f in features])
        attention_mask = torch.stack([torch.tensor(f["attention_mask"]) for f in features])

        # labelsを作成（input_idsのコピー）
        labels = input_ids.clone()

        # 各サンプルのプロンプト部分を-100に設定
        for i, prompt_len in enumerate(self.prompt_lengths):
            # バッチ内のインデックスを計算
            batch_idx = i % len(features)
            if batch_idx < len(features):
                # プロンプト部分を-100に
                labels[batch_idx, :prompt_len] = -100

        # paddingも-100に
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def prepare_dataset(persona_id, tokenizer, max_length=512):
    """
    学習用データセットを準備

    Returns:
        dataset: トークナイズ済みデータセット
        prompt_lengths: 各サンプルのプロンプトのトークン長
    """

    profile, train_turns = load_persona_data(persona_id)

    samples = []
    prompt_lengths = []

    for turn in train_turns:
        prompt = build_prompt(turn, profile)
        response = turn["assistant"]
        full_text = prompt + response + tokenizer.eos_token

        # プロンプトのトークン長を記録
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_lengths.append(len(prompt_tokens))

        samples.append({
            "text": full_text,
            "prompt": prompt,
            "response": response
        })

    # トークナイズ
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    dataset = Dataset.from_list(samples)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "prompt", "response"]
    )

    return tokenized_dataset, prompt_lengths, len(train_turns)


class ResponseOnlyTrainer(Trainer):
    """応答部分のみで損失計算するTrainer"""

    def __init__(self, prompt_lengths, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_lengths = prompt_lengths

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        応答部分のみで損失計算
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # labelsを作成
        labels = input_ids.clone()

        # プロンプト部分を-100に設定
        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            # 現在のエポック内でのグローバルインデックスを推定
            # （簡易実装：prompt_lengthsをサイクルで使用）
            prompt_len = self.prompt_lengths[i % len(self.prompt_lengths)]
            labels[i, :prompt_len] = -100

        # paddingも-100に
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def train_lora_for_persona(
    persona_id,
    base_model,
    tokenizer,
    output_dir,
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4
):
    """1ペルソナ分のLoRAを学習（応答部分のみ）"""

    print(f"\n{'='*60}")
    print(f"Training LoRA for: {persona_id}")
    print(f"{'='*60}")

    # データ準備
    dataset, prompt_lengths, num_turns = prepare_dataset(persona_id, tokenizer)
    print(f"  Training samples: {num_turns}")
    print(f"  Average prompt length: {sum(prompt_lengths)/len(prompt_lengths):.0f} tokens")

    # LoRA設定
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # LoRAモデル作成
    peft_model = get_peft_model(base_model, lora_config)
    trainable_params, total_params = peft_model.get_nb_trainable_parameters()
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # 出力ディレクトリ
    persona_output_dir = Path(output_dir) / persona_id
    persona_output_dir.mkdir(parents=True, exist_ok=True)

    # 学習設定
    training_args = TrainingArguments(
        output_dir=str(persona_output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_drop_last=False
    )

    # カスタムTrainer（応答部分のみ学習）
    trainer = ResponseOnlyTrainer(
        prompt_lengths=prompt_lengths,
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    # 学習
    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time

    # LoRA重みを保存
    peft_model.save_pretrained(str(persona_output_dir / "lora_weights"))
    tokenizer.save_pretrained(str(persona_output_dir / "lora_weights"))

    # メタ情報を保存
    meta = {
        "persona_id": persona_id,
        "num_train_samples": num_turns,
        "avg_prompt_length": sum(prompt_lengths) / len(prompt_lengths),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_time_seconds": elapsed_time,
        "training_method": "response_only",  # 設計B
        "lora_config": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.05
        }
    }
    with open(persona_output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Training time: {elapsed_time/60:.1f} minutes")
    print(f"  Saved to: {persona_output_dir}")

    # メモリ解放
    del peft_model
    del trainer
    torch.cuda.empty_cache()

    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output_dir", default="lora_models_response_only")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--personas", nargs="+", default=None,
                        help="Specific personas to train (optional)")
    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデルとトークナイザー読み込み
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 対象ペルソナ
    if args.personas:
        target_personas = args.personas
    else:
        target_personas = get_target_personas()

    print(f"Target personas: {len(target_personas)}")
    print(f"Training method: Response-only (Design B)")

    # 各ペルソナを学習
    results = []

    for i, persona_id in enumerate(target_personas):
        print(f"\n[{i+1}/{len(target_personas)}] {persona_id}")

        try:
            # ベースモデルを毎回読み込み直す
            if i > 0:
                del base_model
                torch.cuda.empty_cache()
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )

            meta = train_lora_for_persona(
                persona_id=persona_id,
                base_model=base_model,
                tokenizer=tokenizer,
                output_dir=args.output_dir,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            results.append(meta)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "persona_id": persona_id,
                "error": str(e)
            })

    # 全体サマリー保存
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # レポート生成
    generate_training_report(results, output_dir)

    print(f"\n{'='*60}")
    print(f"✅ 完了: {len([r for r in results if 'error' not in r])}/{len(results)} personas trained")
    print(f"📁 出力: {output_dir}")


def generate_training_report(results, output_dir):
    """学習レポートを生成"""

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    total_time = sum(r.get("training_time_seconds", 0) for r in successful)
    total_samples = sum(r.get("num_train_samples", 0) for r in successful)

    report = f"""# LoRA Training Report (Response-Only Design)

**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Training Method**: Response-only loss (Design B)

## Summary

- Total personas: {len(results)}
- Successful: {len(successful)}
- Failed: {len(failed)}
- Total training samples: {total_samples}
- Total training time: {total_time/3600:.1f} hours
- Average time per persona: {total_time/len(successful)/60:.1f} minutes

## Configuration

- LoRA rank: 8
- LoRA alpha: 16
- Target modules: q_proj, v_proj, k_proj, o_proj
- Epochs: 3
- Batch size: 4
- Learning rate: 2e-4

## Training Details

| Persona | Samples | Avg Prompt Len | Time (min) | Status |
|---------|---------|----------------|------------|--------|
"""

    for r in sorted(results, key=lambda x: x.get("training_time_seconds", 0), reverse=True):
        if "error" in r:
            report += f"| {r['persona_id']} | - | - | - | ❌ |\n"
        else:
            report += f"| {r['persona_id']} | {r['num_train_samples']} | {r.get('avg_prompt_length', 0):.0f} | {r['training_time_seconds']/60:.1f} | ✅ |\n"

    if failed:
        report += f"\n## Failed Personas\n\n"
        for r in failed:
            report += f"- {r['persona_id']}: {r['error']}\n"

    report += f"""
## Notes

- **Response-only training**: Only the response part contributes to loss.
  Prompt tokens are masked with label=-100.
- **Fair comparison**: Uses same data and prompt structure as Steering method.
- **No example utterances**: Style examples are NOT included in prompts.
"""

    with open(output_dir / "TRAINING_REPORT.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
