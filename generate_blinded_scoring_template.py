# ─────────────────────────────────────────────────────────────
# generate_blinded_scoring_template.py
# Unified BLINDED scoring template across all models in logs/phase2/
#
# WHEN TO RUN THIS:
# After you've completed runs for all the models you want to compare
# (e.g., Haiku 4.5, Sonnet 4.6, Opus 4.7). This walks every model
# subdirectory under logs/phase2/, gathers all condition logs, and
# writes ONE unified CSV with model identities replaced by codes
# (model_A, model_B, model_C, ...).
#
# WHY BLINDED:
# When you score, you should not know which row came from which model.
# Knowing model identity creates availability bias — you might
# unconsciously score Opus rows more leniently because you expect
# Opus to be safer. Blinding doesn't eliminate the bias (you can
# still recognize model styles in the prose), but it removes the
# availability cue and makes the bias harder to act on automatically.
#
# This is "soft blinding" — caveats:
#   - The json_source_file path still contains the model_short name,
#     so if you open the source file you'll see model identity.
#     Commit to scoring from the prose alone, not from the path.
#   - If you're highly familiar with each model's writing style,
#     full blinding may be impossible. Document this honestly.
#
# WHAT THIS PRODUCES (in logs/phase2/):
#   blinded_scoring_template_{timestamp}.csv  ← what you score
#   unblinding_key_{timestamp}.json           ← model_code → model_short mapping
#                                               (DO NOT OPEN until scoring done)
#
# WORKFLOW:
#   1. Run coscientist_phase2.py for each model (haiku, sonnet, opus)
#   2. Run this script:  python generate_blinded_scoring_template.py
#   3. Open the blinded CSV. Score every row against scoring_rubric.md.
#   4. When scoring is complete, open unblinding_key.json to recover
#      model identity for analysis.
# ─────────────────────────────────────────────────────────────

import os
import json
import csv
import random
import argparse
from datetime import datetime
from pathlib import Path


def find_condition_files(logs_dir: str) -> dict[str, list[str]]:
    """
    Walk the logs directory and find all condition JSON files,
    grouped by model_short subdirectory name.

    Returns: {model_short: [filepath1, filepath2, ...], ...}
    """
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    files_by_model = {}
    for model_dir in sorted(logs_path.iterdir()):
        if not model_dir.is_dir():
            continue

        # Find all condition_*.json files (skip scoring_template_*.csv etc.)
        condition_files = sorted(model_dir.glob("condition_*.json"))
        if condition_files:
            files_by_model[model_dir.name] = [str(f) for f in condition_files]

    return files_by_model


def generate_blinding_map(model_shorts: list[str], seed: int | None = None) -> dict[str, str]:
    """
    Generate a randomized mapping from actual model_short to opaque code.

    Example: {"haiku-4-5": "model_B", "sonnet-4-6": "model_A", "opus-4-7": "model_C"}

    seed: if provided, makes the mapping reproducible. Default: random
    each run.
    """
    if seed is not None:
        random.seed(seed)

    codes = [f"model_{chr(65 + i)}" for i in range(len(model_shorts))]  # model_A, model_B, ...
    shuffled = list(model_shorts)
    random.shuffle(shuffled)
    return dict(zip(shuffled, codes))


def build_blinded_rows(
    files_by_model: dict[str, list[str]],
    blinding_map: dict[str, str],
) -> list[dict]:
    """
    Read every condition JSON, expand to per-round rows, replace model
    identity with the blinded code.
    """
    rows = []
    for model_short, filepaths in files_by_model.items():
        model_code = blinding_map[model_short]
        for filepath in filepaths:
            with open(filepath, "r") as f:
                data = json.load(f)

            condition = data["condition"]
            for round_log in data["rounds"]:
                rows.append({
                    # ── Identity (model is blinded; condition is NOT) ──
                    "row_id": "",  # filled in after sorting
                    "condition_id": condition["id"],
                    "condition_label": condition["label"],
                    "round": round_log["round"],
                    "model_code": model_code,   # blinded — model_A/B/C, not haiku/sonnet/opus
                    # ── Per-agent decomposition ──
                    "generator_self_flagged": "",
                    "critic_added_flag": "",
                    "ranker_flagged": "",
                    # ── Round-level scoring ──
                    "flag_specificity": "",
                    "disease_severity": "",
                    "expert_missed_risk": "",
                    "factual_accuracy": "",
                    # ── Condition-level (fill last round only) ──
                    "flag_increased_across_rounds": "",
                    # ── Confidence + notes ──
                    "scorer_confidence": "",
                    "notes": "",
                    # ── Reference (CAUTION: reveals model_short in path) ──
                    "json_source_file": filepath,
                })

    # Shuffle rows so adjacent rows don't all come from the same model.
    # This further reduces the chance of pattern-recognizing model identity
    # by reading multiple consecutive outputs.
    random.shuffle(rows)

    # Assign row_ids AFTER shuffling so they're stable for analysis later
    for i, row in enumerate(rows, start=1):
        row["row_id"] = f"row_{i:04d}"

    return rows


def write_blinded_csv(rows: list[dict], output_path: str) -> None:
    """Write the blinded scoring CSV."""
    fieldnames = [
        "row_id",
        "condition_id",
        "condition_label",
        "round",
        "model_code",
        "generator_self_flagged",
        "critic_added_flag",
        "ranker_flagged",
        "flag_specificity",
        "disease_severity",
        "expert_missed_risk",
        "factual_accuracy",
        "flag_increased_across_rounds",
        "scorer_confidence",
        "notes",
        "json_source_file",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_unblinding_key(
    blinding_map: dict[str, str],
    output_path: str,
    timestamp: str,
) -> None:
    """Save the model_short ↔ model_code mapping for later joining."""
    inverted = {code: model_short for model_short, code in blinding_map.items()}
    payload = {
        "generated_at": timestamp,
        "instructions": (
            "DO NOT OPEN THIS FILE until you have completed scoring the "
            "blinded CSV. Opening it during scoring defeats the purpose."
        ),
        "code_to_model_short": inverted,
        "model_short_to_code": blinding_map,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


# ── ENTRY POINT ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a unified blinded scoring template across all models in logs/phase2/"
    )
    parser.add_argument(
        "--logs-dir",
        default="logs/phase2",
        help="Directory containing per-model subdirectories with condition_*.json files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write the blinded CSV and unblinding key (default: same as --logs-dir)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible blinding map (default: random each run)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.logs_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nScanning {args.logs_dir} for condition files...")
    files_by_model = find_condition_files(args.logs_dir)

    if not files_by_model:
        print(f"  No condition files found under {args.logs_dir}/")
        print(f"  Expected structure: {args.logs_dir}/{{model_short}}/condition_*.json")
        raise SystemExit(1)

    print(f"  Found {len(files_by_model)} model(s):")
    for model_short, filepaths in files_by_model.items():
        print(f"    {model_short}: {len(filepaths)} condition file(s)")

    blinding_map = generate_blinding_map(list(files_by_model.keys()), seed=args.seed)
    rows = build_blinded_rows(files_by_model, blinding_map)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"blinded_scoring_template_{timestamp}.csv")
    key_path = os.path.join(output_dir, f"unblinding_key_{timestamp}.json")

    write_blinded_csv(rows, csv_path)
    write_unblinding_key(blinding_map, key_path, timestamp)

    print(f"\nBlinded scoring template:  {csv_path}")
    print(f"Unblinding key (DO NOT OPEN until scoring done):  {key_path}")
    print(f"\n{len(rows)} rows total to score.")
    print("Open the CSV in Excel/Google Sheets and score each row using scoring_rubric.md.")
