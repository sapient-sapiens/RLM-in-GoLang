#!/usr/bin/env python3
"""Evaluate OOLONG-Pairs predictions against gold answers.

Usage:
    python3 evaluate.py --preds_dir preds/all_16k --gold_dir data/oolong/gold
"""
import argparse, os, sys


def load_pairs(path):
    """Read a file of (user_id_1, user_id_2) pairs, one per line."""
    pairs = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.strip("() ")
            parts = line.split(",")
            if len(parts) == 2:
                try:
                    a, b = int(parts[0].strip()), int(parts[1].strip())
                    pairs.add((min(a, b), max(a, b)))
                except ValueError:
                    continue
    return pairs


def evaluate(gold_pairs, pred_pairs):
    tp = len(gold_pairs & pred_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "gold_count": len(gold_pairs), "pred_count": len(pred_pairs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_dir", required=True)
    parser.add_argument("--gold_dir", default="data/oolong/gold")
    args = parser.parse_args()

    tasks = []
    for fname in sorted(os.listdir(args.gold_dir)):
        if fname.startswith("task_") and fname.endswith(".txt"):
            task_num = int(fname.replace("task_", "").replace(".txt", ""))
            tasks.append(task_num)

    print(f"{'Task':>6} {'Gold':>6} {'Pred':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-" * 74)

    total_tp = total_fp = total_fn = 0
    evaluated = 0

    for t in sorted(tasks):
        gold_path = os.path.join(args.gold_dir, f"task_{t}.txt")
        pred_path = os.path.join(args.preds_dir, f"task_{t}.txt")

        if not os.path.exists(pred_path):
            print(f"{t:>6} {'—':>6} {'MISS':>6}")
            continue

        gold = load_pairs(gold_path)
        pred = load_pairs(pred_path)
        m = evaluate(gold, pred)

        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]
        evaluated += 1

        print(f"{t:>6} {m['gold_count']:>6} {m['pred_count']:>6} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")

    if evaluated > 0:
        macro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        macro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0
        print("-" * 74)
        print(f"{'MICRO':>6} {'':>6} {'':>6} {total_tp:>6} {total_fp:>6} {total_fn:>6} "
              f"{macro_p:>8.4f} {macro_r:>8.4f} {macro_f1:>8.4f}")
        print(f"\nEvaluated {evaluated}/{len(tasks)} tasks")


if __name__ == "__main__":
    main()
