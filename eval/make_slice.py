"""Generate a deterministic, stratified 250-query slice of BIRD dev set.

Uses random.seed(42) to sample proportionally across simple, moderate,
and challenging difficulty bands.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def make_stratified_slice(
    bird_dev_path: str = "data/benchmarks/bird/dev.json",
    output_path: str = "data/bird_dev_slice_250.json",
    sample_size: int = 250,
    seed: int = 42,
) -> List[Dict]:
    """Sample queries proportionally across difficulty bands."""
    dev_file = Path(bird_dev_path)
    if not dev_file.exists():
        raise FileNotFoundError(f"BIRD dev file not found at {dev_file}")

    with open(dev_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group by difficulty
    by_diff: Dict[str, List[Dict]] = {}
    for item in data:
        diff = item.get("difficulty", "moderate")
        by_diff.setdefault(diff, []).append(item)

    total_items = len(data)
    rng = random.Random(seed)

    # Calculate proportional counts
    # E.g. simple: 925/1534 * 250 = 150.75 -> 151
    # moderate: 464/1534 * 250 = 75.62 -> 76
    # challenging: 145/1534 * 250 = 23.63 -> 23
    sampled: List[Dict] = []
    diff_keys = sorted(by_diff.keys())  # deterministic ordering: challenging, moderate, simple
    
    # Calculate target allocations
    allocations = {}
    allocated_sum = 0
    for diff in diff_keys:
        k = round(len(by_diff[diff]) / total_items * sample_size)
        allocations[diff] = k
        allocated_sum += k

    # Adjust difference if rounding caused +/- 1
    diff_delta = sample_size - allocated_sum
    if diff_delta != 0:
        # adjust largest category
        largest_cat = max(diff_keys, key=lambda d: len(by_diff[d]))
        allocations[largest_cat] += diff_delta

    print(f"Stratified sample allocation (Total {sample_size}):")
    for diff in diff_keys:
        print(f"  {diff}: {allocations[diff]} / {len(by_diff[diff])}")
        items = list(by_diff[diff])
        # Sort by question_id before shuffling with seed for absolute reproducibility
        items.sort(key=lambda x: x["question_id"])
        chosen = rng.sample(items, allocations[diff])
        sampled.extend(chosen)

    # Sort final slice by question_id
    sampled.sort(key=lambda x: x["question_id"])

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {len(sampled)} queries -> {output_path}")
    return sampled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stratified BIRD dev slice.")
    parser.add_argument(
        "--bird-dev",
        default="data/benchmarks/bird/dev.json",
        help="Path to BIRD dev.json",
    )
    parser.add_argument(
        "--output",
        default="data/bird_dev_slice_250.json",
        help="Output path for the 250-query slice",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=250,
        help="Number of queries to sample (default: 250)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    args = parser.parse_args()

    make_stratified_slice(
        bird_dev_path=args.bird_dev,
        output_path=args.output,
        sample_size=args.sample_size,
        seed=args.seed,
    )
