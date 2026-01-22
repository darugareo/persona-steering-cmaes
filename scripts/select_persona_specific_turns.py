#!/usr/bin/env python3
"""
ペルソナ特有のターンのみを選定

GPT-4oを使用して、各ペルソナのtrain/test_turns.jsonから
「ペルソナ特有」のターンのみを抽出する。
"""

import json
import os
from pathlib import Path
from openai import OpenAI
import time
import sys

# OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ Error: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

client = OpenAI(api_key=api_key)


def judge_persona_specificity(turn, persona_profile):
    """ターンがペルソナ特有かどうか判定"""

    speaker_role = persona_profile.get("speaker_role", "Unknown")
    relationship = persona_profile.get("relationship", "Unknown")

    # ターンからデータを取得（異なる形式に対応）
    context = turn.get('context', '')
    user_input = turn.get('user', turn.get('user_message', turn.get('input', '')))
    assistant_response = turn.get('assistant', turn.get('assistant_message', turn.get('ground_truth', '')))

    prompt = f"""あなたは会話データの品質を評価するエキスパートです。

## タスク
以下の会話ターンが「ペルソナ特有」かどうか判定してください。

## ペルソナ情報
- 役割: {speaker_role}
- 関係性: {relationship}

## 会話ターン
Context（文脈）:
{context if context else 'N/A'}

Partner's Input（相手の発話）:
{user_input}

Persona's Response（ペルソナの応答 = Ground Truth）:
{assistant_response}

## 判定基準

「ペルソナ特有」とは：
1. この役割（{speaker_role}）でなければ自然に答えられない
2. 第三者が同じ応答をしたら明らかに不自然
3. 関係性や共有の記憶への言及がある
4. 役割特有の視点や感情が含まれる

「ペルソナ特有ではない」とは：
1. 誰でも同じように答えられる汎用的な応答
2. 一般的な事実や意見
3. ペルソナの視点が不要

## 出力形式（JSON）
{{
    "persona_specific": true/false,
    "confidence": 1-5,
    "reason": "判定理由（日本語、50文字以内）",
    "criteria_met": ["関係性特有", "共有記憶", "役割視点"]
}}

JSONのみを出力してください。"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300
        )

        result_text = response.choices[0].message.content.strip()

        # JSON部分を抽出
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result = json.loads(result_text.strip())
        return result

    except Exception as e:
        print(f"\n  ⚠️ Error: {e}")
        return {
            "persona_specific": False,
            "confidence": 1,
            "reason": f"Error: {str(e)[:50]}",
            "criteria_met": []
        }


def process_persona(persona_id, checkpoint_file=None):
    """1ペルソナのターンを選定"""

    persona_dir = Path(f"personas_cc/{persona_id}")

    if not persona_dir.exists():
        print(f"  ⚠️ Directory not found: {persona_dir}")
        return None

    # Profile読み込み
    profile_file = persona_dir / "persona_profile.txt"
    if not profile_file.exists():
        profile_file = persona_dir / "profile.json"

    if profile_file.exists() and profile_file.suffix == ".json":
        with open(profile_file) as f:
            profile = json.load(f)
    else:
        # テキストプロファイルの場合、基本情報のみ設定
        profile = {
            "speaker_role": "Unknown",
            "relationship": "Unknown"
        }

    results = {
        "train": {"total": 0, "selected": 0, "turns": []},
        "test": {"total": 0, "selected": 0, "turns": []}
    }

    # チェックポイントから再開
    if checkpoint_file and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
            if checkpoint.get("persona_id") == persona_id:
                results = checkpoint.get("results", results)
                print(f"  📥 Checkpoint loaded: {checkpoint.get('last_split')}")

    for split in ["train", "test"]:
        # 既に処理済みの場合はスキップ
        if results[split]["selected"] > 0:
            print(f"  ✅ {split} already processed ({results[split]['selected']} selected)")
            continue

        # ターン読み込み
        turns_file = persona_dir / f"{split}_turns.json"
        if not turns_file.exists():
            # 代替ファイル名を試す
            turns_file = persona_dir / f"{split}_turns_selected.json"

        if not turns_file.exists():
            print(f"  ⚠️ {split}_turns.json not found")
            continue

        with open(turns_file) as f:
            data = json.load(f)

        turns = data.get("turns", data if isinstance(data, list) else [])
        results[split]["total"] = len(turns)

        selected_turns = []

        print(f"\n  Processing {split} turns...")
        for i, turn in enumerate(turns):
            print(f"    Turn {i+1}/{len(turns)}...", end=" ", flush=True)

            judgment = judge_persona_specificity(turn, profile)

            # ターンキーの正規化
            turn_normalized = {
                "user": turn.get('user', turn.get('user_message', turn.get('input', ''))),
                "assistant": turn.get('assistant', turn.get('assistant_message', turn.get('ground_truth', ''))),
                "context": turn.get('context', ''),
                "persona_specific": judgment["persona_specific"],
                "specificity_confidence": judgment["confidence"],
                "specificity_reason": judgment["reason"],
                "criteria_met": judgment.get("criteria_met", [])
            }

            if judgment["persona_specific"] and judgment["confidence"] >= 3:
                selected_turns.append(turn_normalized)
                print(f"✅ Selected (conf={judgment['confidence']})")
            else:
                reason_short = judgment['reason'][:30] + "..." if len(judgment['reason']) > 30 else judgment['reason']
                print(f"❌ Skipped ({reason_short})")

            # チェックポイント保存
            if (i + 1) % 5 == 0 and checkpoint_file:
                results[split]["selected"] = len(selected_turns)
                results[split]["turns"] = selected_turns
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        "persona_id": persona_id,
                        "last_split": split,
                        "last_turn_index": i,
                        "results": results
                    }, f, indent=2)

            time.sleep(0.5)  # Rate limiting

        results[split]["selected"] = len(selected_turns)
        results[split]["turns"] = selected_turns

    return results


def main():
    personas_dir = Path("personas_cc")

    if not personas_dir.exists():
        print(f"❌ Error: {personas_dir} not found")
        sys.exit(1)

    persona_ids = sorted([p.name for p in personas_dir.iterdir()
                          if p.is_dir() and p.name.startswith("episode-")])

    print(f"=" * 80)
    print(f"ペルソナ特有ターン選定")
    print(f"=" * 80)
    print(f"対象ペルソナ数: {len(persona_ids)}")
    print(f"API: OpenAI GPT-4o")
    print(f"=" * 80)

    all_results = {}
    summary = []

    # 出力ディレクトリ
    output_dir = Path("results/persona_specific_selection")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, persona_id in enumerate(persona_ids):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(persona_ids)}] {persona_id}")
        print(f"{'='*80}")

        checkpoint_file = output_dir / f"{persona_id}_checkpoint.json"

        try:
            results = process_persona(persona_id, checkpoint_file)

            if results is None:
                print(f"  ⚠️ Skipped")
                continue

            all_results[persona_id] = results

            # サマリー追加
            summary.append({
                "persona_id": persona_id,
                "train_total": results["train"]["total"],
                "train_selected": results["train"]["selected"],
                "train_rate": results["train"]["selected"] / max(results["train"]["total"], 1),
                "test_total": results["test"]["total"],
                "test_selected": results["test"]["selected"],
                "test_rate": results["test"]["selected"] / max(results["test"]["total"], 1),
            })

            # 選定されたターンを保存
            persona_output_dir = Path(f"personas_cc/{persona_id}")

            for split in ["train", "test"]:
                output_file = persona_output_dir / f"{split}_turns_persona_specific.json"
                with open(output_file, "w") as f:
                    json.dump({
                        "persona_id": persona_id,
                        "split": split,
                        "total_turns": results[split]["total"],
                        "selected_turns": results[split]["selected"],
                        "selection_rate": results[split]["selected"] / max(results[split]["total"], 1),
                        "turns": results[split]["turns"]
                    }, f, indent=2, ensure_ascii=False)

            print(f"\n  📊 Results:")
            print(f"    Train: {results['train']['selected']}/{results['train']['total']} selected ({results['train']['selected']/max(results['train']['total'],1)*100:.0f}%)")
            print(f"    Test: {results['test']['selected']}/{results['test']['total']} selected ({results['test']['selected']/max(results['test']['total'],1)*100:.0f}%)")

            # チェックポイント削除
            if checkpoint_file.exists():
                checkpoint_file.unlink()

        except KeyboardInterrupt:
            print(f"\n\n⚠️ Interrupted by user. Checkpoint saved.")
            break
        except Exception as e:
            print(f"\n  ❌ Error processing {persona_id}: {e}")
            continue

    # 全体サマリーを保存
    with open(output_dir / "selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # サマリーレポート作成
    generate_summary_report(summary, output_dir)

    print(f"\n{'='*80}")
    print(f"✅ 完了: {len(summary)}/{len(persona_ids)} personas processed")
    print(f"📁 結果: {output_dir}")
    print(f"{'='*80}")


def generate_summary_report(summary, output_dir):
    """サマリーレポートを生成"""

    import statistics

    if not summary:
        print("⚠️ No data to generate report")
        return

    train_rates = [s["train_rate"] for s in summary if s["train_total"] > 0]
    test_rates = [s["test_rate"] for s in summary if s["test_total"] > 0]

    report = f"""# ペルソナ特有ターン選定レポート

**生成日**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**対象ペルソナ数**: {len(summary)}

## 全体統計

### Train
- 平均選定率: {statistics.mean(train_rates)*100:.1f}%
- 中央値選定率: {statistics.median(train_rates)*100:.1f}%
- 最小選定率: {min(train_rates)*100:.1f}%
- 最大選定率: {max(train_rates)*100:.1f}%

### Test
- 平均選定率: {statistics.mean(test_rates)*100:.1f}%
- 中央値選定率: {statistics.median(test_rates)*100:.1f}%
- 最小選定率: {min(test_rates)*100:.1f}%
- 最大選定率: {max(test_rates)*100:.1f}%

## ペルソナ別結果

| Persona ID | Train (selected/total) | Test (selected/total) |
|------------|------------------------|----------------------|
"""

    for s in sorted(summary, key=lambda x: x["train_rate"], reverse=True):
        report += f"| {s['persona_id']} | {s['train_selected']}/{s['train_total']} ({s['train_rate']*100:.0f}%) | {s['test_selected']}/{s['test_total']} ({s['test_rate']*100:.0f}%) |\n"

    # 選定率が低いペルソナの警告
    low_selection = [s for s in summary if s["train_rate"] < 0.3 or s["test_rate"] < 0.3]
    if low_selection:
        report += f"\n## ⚠️ 選定率が低いペルソナ（< 30%）\n\n"
        for s in low_selection:
            report += f"- {s['persona_id']}: Train {s['train_rate']*100:.0f}%, Test {s['test_rate']*100:.0f}%\n"

    # ターン数が少ないペルソナ
    low_turns = [s for s in summary if s["train_selected"] < 5 or s["test_selected"] < 5]
    if low_turns:
        report += f"\n## ⚠️ 選定ターン数が少ないペルソナ（< 5）\n\n"
        for s in low_turns:
            report += f"- {s['persona_id']}: Train {s['train_selected']}, Test {s['test_selected']}\n"

    with open(output_dir / "SELECTION_REPORT.md", "w") as f:
        f.write(report)

    print(f"\n📄 Report saved: {output_dir / 'SELECTION_REPORT.md'}")


if __name__ == "__main__":
    main()
