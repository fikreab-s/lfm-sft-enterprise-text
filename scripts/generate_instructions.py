"""
Domain Instruction Data Generator

Generates diverse instruction-response pairs for enterprise SFT training.
Categories: structured output, analytics explanation, scenario planning,
data extraction, and conversation.

Author: Fab Admasu
License: MIT
"""

import json
import random
import argparse
from pathlib import Path

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BRANDS = ["Cardivex", "Immunolex", "OncoPrime", "NeuraStar",
          "RespiClear", "DermaShield", "VaxGuard", "EndoBalance"]

METRICS = ["ROI", "ROAS", "CPM", "CTR", "conversion rate", "response rate",
           "market share", "TRx volume", "NBRx volume", "adherence rate"]

CHANNELS = ["TV", "Digital", "Print", "Email", "Rep Visits", "Webinars",
            "Conferences", "Samples", "Social Media", "Programmatic Display"]


def gen_json_extraction():
    """Generate JSON extraction instruction pairs."""
    brand = random.choice(BRANDS)
    channels = random.sample(CHANNELS, k=random.randint(3, 5))
    data = {}
    for ch in channels:
        data[ch] = {
            "spend_m": round(random.uniform(0.5, 15), 2),
            "roi": round(random.uniform(0.5, 7.0), 2),
            "yoy_change_pct": round(random.uniform(-20, 40), 1),
        }

    instruction = (f"Extract the channel performance data for {brand} "
                   f"and return as structured JSON with spend, ROI, and YoY change.")
    response = json.dumps({"brand": brand, "channels": data}, indent=2)
    return instruction, response, "json_extraction"


def gen_comparison():
    """Generate channel comparison instruction pairs."""
    brand = random.choice(BRANDS)
    ch1, ch2 = random.sample(CHANNELS, 2)
    roi1 = round(random.uniform(1.0, 6.0), 2)
    roi2 = round(random.uniform(0.5, 5.0), 2)
    spend1 = round(random.uniform(1, 10), 1)
    spend2 = round(random.uniform(1, 10), 1)

    instruction = (f"Compare {ch1} vs {ch2} performance for {brand}. "
                   f"{ch1}: spend=${spend1}M, ROI={roi1}x. "
                   f"{ch2}: spend=${spend2}M, ROI={roi2}x.")

    better = ch1 if roi1 > roi2 else ch2
    worse = ch2 if roi1 > roi2 else ch1
    better_roi = max(roi1, roi2)
    worse_roi = min(roi1, roi2)

    response = (f"**{brand} Channel Comparison:**\n\n"
                f"- **{better}** outperforms with {better_roi}x ROI vs "
                f"{worse}'s {worse_roi}x\n"
                f"- ROI differential: {abs(roi1-roi2):.2f}x\n"
                f"- **Recommendation:** Consider shifting budget from "
                f"{worse} to {better}, monitoring saturation effects\n"
                f"- **Caveat:** ROI comparison assumes similar measurement "
                f"windows and attribution methodology")
    return instruction, response, "comparison"


def gen_scenario():
    """Generate scenario planning instruction pairs."""
    brand = random.choice(BRANDS)
    channel = random.choice(CHANNELS)
    change_pct = random.choice([10, 15, 20, 25, -10, -15, -20])
    current_spend = round(random.uniform(2, 15), 1)
    roi = round(random.uniform(1.0, 5.0), 2)

    direction = "increasing" if change_pct > 0 else "decreasing"
    new_spend = round(current_spend * (1 + change_pct/100), 2)
    # Diminishing returns for increases, proportional for decreases
    if change_pct > 0:
        impact_mult = change_pct / 100 * roi * 0.8  # 80% efficiency at margin
    else:
        impact_mult = change_pct / 100 * roi * 1.1  # 110% loss at margin

    instruction = (f"What is the expected impact of {direction} {brand}'s "
                   f"{channel} budget by {abs(change_pct)}%? "
                   f"Current spend: ${current_spend}M, ROI: {roi}x.")

    response = (f"## Scenario: {brand} {channel} Budget "
                f"{'Increase' if change_pct > 0 else 'Decrease'}\n\n"
                f"| Metric | Current | Projected |\n"
                f"|--------|---------|----------|\n"
                f"| Spend | ${current_spend}M | ${new_spend}M |\n"
                f"| Incremental Revenue | ${current_spend * roi:.1f}M | "
                f"${new_spend * roi * (0.95 if change_pct > 0 else 1.0):.1f}M |\n"
                f"| Marginal ROI | {roi}x | "
                f"{roi * (0.85 if change_pct > 0 else 1.05):.2f}x |\n\n"
                f"**Note:** {'Diminishing returns expected at higher spend levels. '
                if change_pct > 0 else 'Budget reduction may have delayed carryover effects. '}"
                f"Monitor for 2-3 quarters to capture full adstock effect.")
    return instruction, response, "scenario"


def gen_sql_query():
    """Generate Text-to-SQL instruction pairs."""
    brand = random.choice(BRANDS)
    metric = random.choice(METRICS)
    channel = random.choice(CHANNELS)

    templates = [
        (f"What was the total {metric} for {brand} {channel} in Q3 2024?",
         f"SELECT SUM({metric.replace(' ', '_')}) AS total_{metric.replace(' ', '_')}\n"
         f"FROM promotional_performance\n"
         f"WHERE brand = '{brand}'\n"
         f"  AND channel = '{channel}'\n"
         f"  AND quarter = 'Q3 2024';"),
        (f"Show me the top 5 channels by {metric} for {brand}.",
         f"SELECT channel, {metric.replace(' ', '_')}\n"
         f"FROM promotional_performance\n"
         f"WHERE brand = '{brand}'\n"
         f"ORDER BY {metric.replace(' ', '_')} DESC\n"
         f"LIMIT 5;"),
        (f"Compare {brand}'s {metric} across all quarters.",
         f"SELECT quarter, channel, AVG({metric.replace(' ', '_')}) AS avg_{metric.replace(' ', '_')}\n"
         f"FROM promotional_performance\n"
         f"WHERE brand = '{brand}'\n"
         f"GROUP BY quarter, channel\n"
         f"ORDER BY quarter, avg_{metric.replace(' ', '_')} DESC;"),
    ]
    instruction, response = random.choice(templates)
    return instruction, response, "text_to_sql"


GENERATORS = [gen_json_extraction, gen_comparison, gen_scenario, gen_sql_query]


def generate_dataset(n_examples: int, output_dir: Path):
    """Generate diverse instruction-response dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    for _ in range(n_examples):
        gen = random.choice(GENERATORS)
        instruction, response, category = gen()
        examples.append({
            "instruction": instruction,
            "response": response,
            "category": category,
        })

    random.shuffle(examples)
    split = int(len(examples) * 0.9)

    for name, data in [("train", examples[:split]), ("eval", examples[split:])]:
        path = output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")

    cat_counts = {}
    for ex in examples:
        cat_counts[ex["category"]] = cat_counts.get(ex["category"], 0) + 1

    print(f"✅ Generated {split} train + {len(examples) - split} eval examples")
    print(f"   Categories: {cat_counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_examples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()
    generate_dataset(args.n_examples, Path(args.output_dir))


if __name__ == "__main__":
    main()
