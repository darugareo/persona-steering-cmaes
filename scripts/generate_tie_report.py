#!/usr/bin/env python3
"""
Generate detailed analysis report for Base vs Steering comparison
Focuses on tie cases and why steering shows limited effect
"""
import json
from pathlib import Path
from collections import defaultdict

def main():
    # Load data
    with open("results/base_vs_steering/comparison_results.json") as f:
        data = json.load(f)
    with open("results/base_vs_steering/tie_analysis.json") as f:
        analysis = json.load(f)

    results = data["results"]

    # Create detailed report
    report_lines = [
        "# Base vs Steering - 詳細分析レポート",
        "",
        "**生成日**: 2026-01-07",
        "**総比較数**: 728 (26 personas × 28 turns)",
        "",
        "---",
        "",
        "## 1. 全体サマリー",
        "",
        "### 判定結果の内訳",
        "- **引き分け (Tie)**: 657件 (90.2%)",
        "- **Steering勝利**: 57件 (7.8%)",
        "- **Base勝利**: 14件 (1.9%)",
        "",
        "### Persona別の傾向",
        "- **全て引き分けのpersona**: 22/26 (84.6%)",
        "- **差が認められたpersona**: 4/26 (15.4%)",
        "",
        "**結論**: 大多数のpersonaでは、SteeringとBaseの出力に識別可能な差が認められない。",
        "",
        "---",
        "",
        "## 2. 全て引き分けだったPersonas (22件)",
        "",
        "以下のpersonasでは、28ターン全てで「両者に差なし」と判定されました：",
        ""
    ]

    for persona_id in sorted(analysis["all_tie_personas"]):
        report_lines.append(f"- {persona_id}")

    report_lines.extend([
        "",
        "### 典型的な引き分けの理由",
        "",
        "Judge (GPT-4o) の説明を分析すると、以下のパターンが見られます：",
        ""
    ])

    # Collect tie explanations
    ties = [r for r in results if r["winner"] == "tie"]
    tie_explanations = defaultdict(int)

    for t in ties[:100]:  # Sample first 100
        exp = t.get("explanation", "")
        if "formal" in exp.lower() and "informal" in exp.lower():
            tie_explanations["Both too formal/informal"] += 1
        elif "deviate" in exp.lower() or "not align" in exp.lower():
            tie_explanations["Both deviate from persona"] += 1
        elif "equally" in exp.lower():
            tie_explanations["Equally (mis)aligned"] += 1
        elif "neither" in exp.lower():
            tie_explanations["Neither matches well"] += 1
        else:
            tie_explanations["Other"] += 1

    for reason, count in sorted(tie_explanations.items(), key=lambda x: x[1], reverse=True):
        report_lines.append(f"- **{reason}**: {count}件")

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. 差が認められたPersonas (4件)",
        "",
        "以下の4 personasでのみ、BaseとSteeringに明確な差が認められました：",
        ""
    ])

    # Detailed analysis for mixed personas
    for persona_id in sorted(analysis["mixed_personas"]):
        persona_results = [r for r in results if r["persona_id"] == persona_id]

        ties_count = len([r for r in persona_results if r["winner"] == "tie"])
        steering_count = len([r for r in persona_results if r["winner"] == "steering"])
        base_count = len([r for r in persona_results if r["winner"] == "base"])
        total = len(persona_results)

        report_lines.extend([
            f"### {persona_id}",
            "",
            "**結果内訳**:",
            f"- 引き分け: {ties_count}/{total} ({100*ties_count/total:.1f}%)",
            f"- Steering勝利: {steering_count}/{total} ({100*steering_count/total:.1f}%)",
            f"- Base勝利: {base_count}/{total} ({100*base_count/total:.1f}%)",
            ""
        ])

    report_lines.extend([
        "---",
        "",
        "## 4. 具体例: 引き分けケース",
        "",
        "以下は引き分けと判定された典型例です。",
        ""
    ])

    # Show 3 tie examples
    for i, tie_item in enumerate(ties[:3], 1):
        prompt_text = str(tie_item.get('prompt', ''))[:200]
        base_text = str(tie_item.get('response_base', ''))[:300]
        steering_text = str(tie_item.get('response_steering', ''))[:300]
        confidence = tie_item.get('confidence', 'N/A')
        explanation = tie_item.get('explanation', 'No explanation')

        report_lines.extend([
            f"### 例{i}: {tie_item['persona_id']}",
            "",
            "**プロンプト**:",
            "```",
            prompt_text + "...",
            "```",
            "",
            "**Base生成**:",
            "```",
            base_text + "...",
            "```",
            "",
            "**Steering生成**:",
            "```",
            steering_text + "...",
            "```",
            "",
            f"**Judge判定**: Tie (信頼度: {confidence}/5)",
            "",
            "**Judge説明**:",
            f"> {explanation}",
            "",
            "---",
            ""
        ])

    # Show steering wins
    steering_wins = [r for r in results if r["winner"] == "steering"]
    report_lines.extend([
        "## 5. 具体例: Steering勝利ケース",
        "",
        "SteeringがBaseを上回ったケースを示します。",
        ""
    ])

    for i, win in enumerate(steering_wins[:3], 1):
        prompt_text = str(win.get('prompt', ''))[:200]
        base_text = str(win.get('response_base', ''))[:300]
        steering_text = str(win.get('response_steering', ''))[:300]
        confidence = win.get('confidence', 'N/A')
        explanation = win.get('explanation', 'No explanation')

        report_lines.extend([
            f"### 例{i}: {win['persona_id']}",
            "",
            "**プロンプト**:",
            "```",
            prompt_text + "...",
            "```",
            "",
            "**Base生成**:",
            "```",
            base_text + "...",
            "```",
            "",
            "**Steering生成**:",
            "```",
            steering_text + "...",
            "```",
            "",
            f"**Judge判定**: Steering勝利 (信頼度: {confidence}/5)",
            "",
            "**Judge説明**:",
            f"> {explanation}",
            "",
            "---",
            ""
        ])

    # Show base wins
    base_wins = [r for r in results if r["winner"] == "base"]
    report_lines.extend([
        "## 6. 具体例: Base勝利ケース",
        "",
        "BaseがSteeringを上回った稀なケースを示します。",
        ""
    ])

    for i, win in enumerate(base_wins[:3], 1):
        prompt_text = str(win.get('prompt', ''))[:200]
        base_text = str(win.get('response_base', ''))[:300]
        steering_text = str(win.get('response_steering', ''))[:300]
        confidence = win.get('confidence', 'N/A')
        explanation = win.get('explanation', 'No explanation')

        report_lines.extend([
            f"### 例{i}: {win['persona_id']}",
            "",
            "**プロンプト**:",
            "```",
            prompt_text + "...",
            "```",
            "",
            "**Base生成**:",
            "```",
            base_text + "...",
            "```",
            "",
            "**Steering生成**:",
            "```",
            steering_text + "...",
            "```",
            "",
            f"**Judge判定**: Base勝利 (信頼度: {confidence}/5)",
            "",
            "**Judge説明**:",
            f"> {explanation}",
            "",
            "---",
            ""
        ])

    report_lines.extend([
        "## 7. 考察",
        "",
        "### 7.1 なぜ90%が引き分けなのか？",
        "",
        "1. **生成の質的類似性**: BaseとSteeringの出力が、Judgeにとって区別不可能なほど似ている",
        "2. **Persona特性の曖昧性**: 多くのpersonaで、固有の文体・語彙が弱く、差を検出しにくい",
        "3. **Steeringの効果が微弱**: Layer 20でのactivation steeringが出力に十分な変化をもたらしていない",
        "4. **Judge基準の問題**: GPT-4oが微細な文体差を検出できていない可能性",
        "",
        "### 7.2 差が認められた4 personasの特徴",
        "",
        "差が認められたpersonasを分析すると：",
        "- **episode-118328_B**: Steering 17勝 vs Base 3勝 (明確な差)",
        "- **episode-184019_A**: Steering 21勝 vs Base 3勝 (圧倒的差)",
        "- **episode-225888_A**: Steering 7勝 vs Base 6勝 (拮抗)",
        "- **episode-239427_A**: Steering 12勝 vs Base 2勝 (明確な差)",
        "",
        "これらのpersonasは、**強い文体的特徴**を持っている可能性が高い。",
        "",
        "### 7.3 4条件実験との矛盾",
        "",
        "**4条件実験** (2 personas × 10 turns = 20比較):",
        "- Steering vs Base: Base 60% vs Steering 40%",
        "",
        "**今回の大規模検証** (26 personas × 28 turns = 728比較):",
        "- 引き分け除外: Steering 80.3% vs Base 19.7%",
        "",
        "**矛盾の原因**:",
        "1. **サンプルサイズ**: 20 vs 728の差",
        "2. **Persona選択バイアス**: 4条件実験の2 personasがたまたまSteering不利だった可能性",
        "3. **評価方法の違い**: Judge promptや設定の微妙な違い",
        "4. **引き分けの扱い**: 4条件実験では引き分けを許さず強制選択していた可能性",
        "",
        "### 7.4 実用的インプリケーション",
        "",
        "**結論**:",
        "- **84.6%のpersonasでSteeringは無効** (全て引き分け)",
        "- **15.4%のpersonasでSteeringは有効** (明確な差あり)",
        "- 大規模検証により、**Steeringの効果は極めて限定的**であることが判明",
        "",
        "**推奨**:",
        "1. Steering効果が現れるpersonaの特徴を分析",
        "2. より強力なsteering手法の検討 (異なるlayer、より大きなalpha)",
        "3. Prompt engineeringとの組み合わせ最適化",
        "",
        "---",
        "",
        "## データファイル",
        "",
        "- 全結果: `results/base_vs_steering/comparison_results.json`",
        "- サマリー: `results/base_vs_steering/summary.json`",
        "- 分析データ: `results/base_vs_steering/tie_analysis.json`",
        ""
    ])

    # Save report
    report_content = "\n".join(report_lines)
    with open("results/base_vs_steering/DETAILED_ANALYSIS.md", "w") as f:
        f.write(report_content)

    print("✓ Saved detailed analysis to results/base_vs_steering/DETAILED_ANALYSIS.md")
    print(f"  Report length: {len(report_content)} characters")
    print(f"  Lines: {len(report_lines)}")

if __name__ == "__main__":
    main()
