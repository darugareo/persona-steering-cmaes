#!/usr/bin/env python3
# scripts/train_lora_per_persona.py

import os
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse

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
    """新しいプロンプト構造でプロンプトを生成"""

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

Your reply:"""

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


def prepare_dataset(persona_id, tokenizer, max_length=512):
    """学習用データセットを準備"""

    profile, train_turns = load_persona_data(persona_id)

    texts = []
    for turn in train_turns:
        prompt = build_prompt(turn, profile)
        # プロンプト + 正解応答
        full_text = prompt + turn["assistant"] + tokenizer.eos_token
        texts.append(full_text)

    # トークナイズ
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    return tokenized_dataset, len(train_turns)


def train_lora_for_persona(
    persona_id,
    base_model,
    tokenizer,
    output_dir,
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4
):
    """1ペルソナ分のLoRAを学習"""

    print(f"\n{'='*60}")
    print(f"Training LoRA for: {persona_id}")
    print(f"{'='*60}")

    # データ準備
    dataset, num_turns = prepare_dataset(persona_id, tokenizer)
    print(f"  Training samples: {num_turns}")

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
    peft_model.print_trainable_parameters()

    # 学習設定
    persona_output_dir = Path(output_dir) / persona_id
    persona_output_dir.mkdir(parents=True, exist_ok=True)

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
        remove_unused_columns=False
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # 学習
    import time
    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time

    # LoRA重みを保存
    peft_model.save_pretrained(str(persona_output_dir / "lora_weights"))

    # メタ情報を保存
    meta = {
        "persona_id": persona_id,
        "num_train_samples": num_turns,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_time_seconds": elapsed_time,
        "lora_config": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
    }
    with open(persona_output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Training time: {elapsed_time/60:.1f} minutes")
    print(f"  Saved to: {persona_output_dir}")

    # メモリ解放のためモデルを削除
    del peft_model
    torch.cuda.empty_cache()

    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output_dir", default="lora_models")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--personas", nargs="+", default=None,
                        help="Specific personas to train (optional)")
    args = parser.parse_args()

    # モデルとトークナイザー読み込み
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
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

    # 各ペルソナを学習
    results = []
    for i, persona_id in enumerate(target_personas):
        print(f"\n[{i+1}/{len(target_personas)}] {persona_id}")

        try:
            # ベースモデルを毎回読み込み直す（LoRA重みをリセット）
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
            results.append({
                "persona_id": persona_id,
                "error": str(e)
            })

    # 全体サマリー保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # レポート生成
    generate_training_report(results, output_dir)

    print(f"\n✅ 完了: {len(results)} personas trained")
    print(f"📁 出力: {output_dir}")


def generate_training_report(results, output_dir):
    """学習レポートを生成"""

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    total_time = sum(r.get("training_time_seconds", 0) for r in successful)

    report = f"""# LoRA Training Report

**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- Total personas: {len(results)}
- Successful: {len(successful)}
- Failed: {len(failed)}
- Total training time: {total_time/3600:.1f} hours

## Training Details

| Persona | Samples | Time (min) | Status |
|---------|---------|------------|--------|
"""

    for r in results:
        if "error" in r:
            report += f"| {r['persona_id']} | - | - | ❌ {r['error'][:30]}... |\n"
        else:
            report += f"| {r['persona_id']} | {r['num_train_samples']} | {r['training_time_seconds']/60:.1f} | ✅ |\n"

    if failed:
        report += f"\n## Failed Personas\n\n"
        for r in failed:
            report += f"- {r['persona_id']}: {r['error']}\n"

    with open(output_dir / "TRAINING_REPORT.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
