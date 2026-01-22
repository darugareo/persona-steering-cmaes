#!/usr/bin/env python3
"""
ペルソナ特有ターン選定結果のサマリー生成

選定結果を集計し、詳細な報告書を作成する
"""

import json
from pathlib import Path
from datetime import datetime

def main():
    personas_dir = Path("personas_cc")

    # 全ペルソナを取得
    persona_ids = sorted([d.name for d in personas_dir.iterdir() if d.is_dir()])

    results = []
    total_train_selected = 0
    total_train_total = 0
    total_test_selected = 0
    total_test_total = 0

    print("=" * 80)
    print("ペルソナ特有ターン選定 - 結果サマリー")
    print("=" * 80)
    print(f"対象ペルソナ数: {len(persona_ids)}")
    print("=" * 80)
    print()

    for persona_id in persona_ids:
        persona_dir = personas_dir / persona_id

        train_file = persona_dir / "train_turns_persona_specific.json"
        test_file = persona_dir / "test_turns_persona_specific.json"

        if not train_file.exists() or not test_file.exists():
            print(f"⚠️  {persona_id}: 選定ファイルが見つかりません")
            continue

        # Train結果
        with open(train_file) as f:
            train_data = json.load(f)

        # Test結果
        with open(test_file) as f:
            test_data = json.load(f)

        train_selected = train_data['selected_turns']
        train_total = train_data['total_turns']
        train_rate = train_data['selection_rate']

        test_selected = test_data['selected_turns']
        test_total = test_data['total_turns']
        test_rate = test_data['selection_rate']

        results.append({
            'persona_id': persona_id,
            'train_selected': train_selected,
            'train_total': train_total,
            'train_rate': train_rate,
            'test_selected': test_selected,
            'test_total': test_total,
            'test_rate': test_rate
        })

        total_train_selected += train_selected
        total_train_total += train_total
        total_test_selected += test_selected
        total_test_total += test_total

        # 警告チェック
        warnings = []
        if train_selected < 5:
            warnings.append(f"Train少ない({train_selected})")
        if test_selected < 5:
            warnings.append(f"Test少ない({test_selected})")

        status = "⚠️ " if warnings else "✅"
        warning_str = f" [{', '.join(warnings)}]" if warnings else ""

        print(f"{status} {persona_id:20s} | Train: {train_selected:2d}/{train_total:2d} ({train_rate*100:4.0f}%) | Test: {test_selected:2d}/{test_total:2d} ({test_rate*100:4.0f}%){warning_str}")

    print()
    print("=" * 80)
    print("全体統計")
    print("=" * 80)

    avg_train_rate = total_train_selected / total_train_total if total_train_total > 0 else 0
    avg_test_rate = total_test_selected / total_test_total if total_test_total > 0 else 0

    print(f"Train: {total_train_selected}/{total_train_total} 選定 (平均選定率: {avg_train_rate*100:.1f}%)")
    print(f"Test:  {total_test_selected}/{total_test_total} 選定 (平均選定率: {avg_test_rate*100:.1f}%)")
    print()

    # ペルソナあたりの平均
    avg_train_per_persona = total_train_selected / len(results) if results else 0
    avg_test_per_persona = total_test_selected / len(results) if results else 0

    print(f"ペルソナあたり平均:")
    print(f"  Train: {avg_train_per_persona:.1f} ターン")
    print(f"  Test:  {avg_test_per_persona:.1f} ターン")
    print()

    # 問題のあるペルソナ
    problem_personas = [r for r in results if r['train_selected'] < 5 or r['test_selected'] < 5]

    if problem_personas:
        print("=" * 80)
        print(f"⚠️  選定ターン数が少ないペルソナ ({len(problem_personas)}個)")
        print("=" * 80)
        for r in problem_personas:
            print(f"  {r['persona_id']:20s} | Train: {r['train_selected']:2d} | Test: {r['test_selected']:2d}")
        print()

    # 成功基準チェック
    print("=" * 80)
    print("成功基準チェック")
    print("=" * 80)

    criteria_met = []

    # 1. 各ペルソナで Train 5ターン以上、Test 5ターン以上選定
    personas_sufficient = sum(1 for r in results if r['train_selected'] >= 5 and r['test_selected'] >= 5)
    criterion1 = personas_sufficient == len(results)
    criteria_met.append(criterion1)
    print(f"{'✅' if criterion1 else '❌'} 各ペルソナで Train 5以上、Test 5以上: {personas_sufficient}/{len(results)} ペルソナ")

    # 2. 平均選定率 30%以上
    criterion2 = avg_train_rate >= 0.3 and avg_test_rate >= 0.3
    criteria_met.append(criterion2)
    print(f"{'✅' if criterion2 else '❌'} 平均選定率 30%以上: Train {avg_train_rate*100:.1f}%, Test {avg_test_rate*100:.1f}%")

    print()

    if all(criteria_met):
        print("🎉 全ての成功基準を満たしています！")
        print("   次のフェーズ（再最適化）に進むことができます。")
    else:
        print("⚠️  一部の基準を満たしていません。")
        if not criterion1:
            print("   → 対策: 選定基準を緩和するか、手動でターンを追加")
        if not criterion2:
            print("   → 対策: 選定基準を見直すか、元データを確認")

    print()
    print("=" * 80)

    # JSON形式で保存
    results_dir = Path("results/persona_specific_selection")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_file = results_dir / "selection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_personas': len(results),
            'total_train_selected': total_train_selected,
            'total_train_total': total_train_total,
            'avg_train_rate': avg_train_rate,
            'total_test_selected': total_test_selected,
            'total_test_total': total_test_total,
            'avg_test_rate': avg_test_rate,
            'avg_train_per_persona': avg_train_per_persona,
            'avg_test_per_persona': avg_test_per_persona,
            'problem_personas': len(problem_personas),
            'criteria_met': all(criteria_met),
            'results': results
        }, f, indent=2)

    print(f"📁 サマリー保存: {summary_file}")

    # Markdown形式のレポート作成
    report_file = results_dir / "SELECTION_REPORT.md"
    with open(report_file, 'w') as f:
        f.write("# ペルソナ特有ターン選定 - 実行結果レポート\n\n")
        f.write(f"**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## 📊 全体統計\n\n")
        f.write(f"- **対象ペルソナ数**: {len(results)}\n")
        f.write(f"- **Train選定率**: {avg_train_rate*100:.1f}% ({total_train_selected}/{total_train_total} ターン)\n")
        f.write(f"- **Test選定率**: {avg_test_rate*100:.1f}% ({total_test_selected}/{total_test_total} ターン)\n")
        f.write(f"- **ペルソナあたり平均**: Train {avg_train_per_persona:.1f}ターン, Test {avg_test_per_persona:.1f}ターン\n\n")

        f.write("## ✅ 成功基準\n\n")
        f.write(f"1. {'✅' if criterion1 else '❌'} 各ペルソナで Train 5以上、Test 5以上: {personas_sufficient}/{len(results)}\n")
        f.write(f"2. {'✅' if criterion2 else '❌'} 平均選定率 30%以上: Train {avg_train_rate*100:.1f}%, Test {avg_test_rate*100:.1f}%\n\n")

        if all(criteria_met):
            f.write("**結果**: 🎉 全ての基準を満たしています！\n\n")
        else:
            f.write("**結果**: ⚠️ 一部の基準を満たしていません\n\n")

        f.write("---\n\n")

        f.write("## 📋 ペルソナ別詳細\n\n")
        f.write("| ペルソナID | Train選定 | Train率 | Test選定 | Test率 | 状態 |\n")
        f.write("|------------|-----------|---------|----------|--------|------|\n")

        for r in results:
            status = "⚠️" if r['train_selected'] < 5 or r['test_selected'] < 5 else "✅"
            f.write(f"| {r['persona_id']} | {r['train_selected']}/{r['train_total']} | {r['train_rate']*100:.0f}% | {r['test_selected']}/{r['test_total']} | {r['test_rate']*100:.0f}% | {status} |\n")

        f.write("\n---\n\n")

        if problem_personas:
            f.write("## ⚠️ 注意が必要なペルソナ\n\n")
            f.write(f"選定ターン数が5未満のペルソナ: {len(problem_personas)}個\n\n")
            for r in problem_personas:
                f.write(f"- **{r['persona_id']}**: Train {r['train_selected']}ターン, Test {r['test_selected']}ターン\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("## 📝 次のステップ\n\n")

        if all(criteria_met):
            f.write("### ✅ Phase 2: 再最適化に進む\n\n")
            f.write("```bash\n")
            f.write("# 1ペルソナでテスト\n")
            f.write("python3 scripts/optimize_with_persona_specific.py --persona_id episode-184019_A\n\n")
            f.write("# 全ペルソナで実行\n")
            f.write("python3 scripts/run_all_persona_specific_optimization.py\n")
            f.write("```\n\n")
        else:
            f.write("### ⚠️ 問題の対処が必要\n\n")
            if not criterion1:
                f.write("**選定ターン数不足**:\n")
                f.write("- 選定基準を緩和 (confidence >= 2)\n")
                f.write("- 元のターン数を増やす\n")
                f.write("- 手動で追加ターンを作成\n\n")
            if not criterion2:
                f.write("**選定率が低い**:\n")
                f.write("- 元データの品質確認\n")
                f.write("- 判定基準の見直し\n\n")

    print(f"📄 レポート保存: {report_file}")
    print()
    print("=" * 80)
    print("✅ サマリー生成完了")
    print("=" * 80)
    print()
    print(f"レポート確認: cat {report_file}")
    print()

if __name__ == "__main__":
    main()
