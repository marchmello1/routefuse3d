from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data import structure_category


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument(
        "--cfos-dense",
        action="store_true",
        help="Sensitivity audit: group c-Fos with isolated-dense instead of isolated-sparse",
    )
    args = parser.parse_args()

    def category_for(record: dict) -> str:
        if args.cfos_dense and record["subtype"] == "cfos":
            return "isolated_dense"
        return structure_category(record["subtype"])

    records = []
    for path in sorted(args.cache.glob("fold*_metrics.json")):
        records.extend(json.loads(path.read_text()))
    if not records:
        raise RuntimeError("No fold metrics found")
    route_by_case = {record["case"]: record.get("route_correct") for record in records}
    known_routes = [value for value in route_by_case.values() if value is not None]
    route_examples = {
        record["case"]: (float(record["route_score"]), bool(record["contiguous"]))
        for record in records
        if "route_score" in record
    }
    route_calibration = None
    if route_examples:
        choices = []
        for threshold in np.linspace(0.05, 0.95, 19):
            correct = [
                (score >= threshold) == contiguous
                for score, contiguous in route_examples.values()
            ]
            choices.append((float(np.mean(correct)), float(threshold)))
        accuracy, threshold = max(choices, key=lambda item: (item[0], -abs(item[1] - 0.5)))
        route_calibration = {"threshold": threshold, "accuracy": accuracy}
    density_examples = {
        record["case"]: (
            float(record["density_score"]),
            category_for(record).endswith("_dense"),
        )
        for record in records
        if record.get("density_score") is not None
    }
    density_calibration = None
    if density_examples:
        choices = []
        for threshold in np.linspace(0.05, 0.95, 19):
            correct = [
                (score >= threshold) == dense
                for score, dense in density_examples.values()
            ]
            choices.append((float(np.mean(correct)), float(threshold)))
        accuracy, threshold = max(
            choices, key=lambda item: (item[0], -abs(item[1] - 0.5))
        )
        density_calibration = {"threshold": threshold, "accuracy": accuracy}
    if known_routes:
        print(f"route_accuracy={np.mean(known_routes):.4f} n={len(known_routes)}")
    groups = defaultdict(list)
    for record in records:
        category = category_for(record)
        key = (
            category,
            record["subtype"],
            record.get("shared_weight", 0.65),
            record["threshold"],
            record["minimum_size"],
        )
        groups[key].append(record)
    summaries = []
    for key, group in groups.items():
        topology_key = "cldice" if group[0]["contiguous"] else "pq"
        summaries.append(
            {
                "category": key[0],
                "subtype": key[1],
                "shared_weight": key[2],
                "threshold": key[3],
                "minimum_size": key[4],
                "n": len(group),
                "dice": float(np.mean([x["dice"] for x in group])),
                topology_key: float(np.mean([x[topology_key] for x in group])),
                "objective": float(np.mean([(x["dice"] + x[topology_key]) / 2 for x in group])),
            }
        )
    best = {}
    for summary in summaries:
        subtype = summary["subtype"]
        if subtype not in best or summary["objective"] > best[subtype]["objective"]:
            best[subtype] = summary
    configurations = defaultdict(list)
    for record in records:
        is_contiguous = bool(record["contiguous"])
        key = (
            is_contiguous,
            record.get("shared_weight", 0.65),
            record["threshold"],
            record["minimum_size"],
        )
        configurations[key].append(record)
    recommended = {}
    category_choices = defaultdict(list)
    for contiguous in (False, True):
        choices = []
        for key, values in configurations.items():
            if key[0] != contiguous:
                continue
            matching = values
            topology_key = "cldice" if contiguous else "pq"
            by_category = defaultdict(list)
            for item in matching:
                category = category_for(item)
                by_category[category].append(item)
            category_metrics = {
                category: {
                    "dice": float(np.mean([item["dice"] for item in items])),
                    topology_key: float(np.mean([item[topology_key] for item in items])),
                    "objective": float(
                        np.mean(
                            [(item["dice"] + item[topology_key]) / 2 for item in items]
                        )
                    ),
                }
                for category, items in by_category.items()
            }
            for category, metrics in category_metrics.items():
                category_choices[category].append(
                    {
                        "shared_weight": key[1],
                        "threshold": key[2],
                        "minimum_size": key[3],
                        **metrics,
                    }
                )
            choices.append(
                {
                    "shared_weight": key[1],
                    "threshold": key[2],
                    "minimum_size": key[3],
                    "macro_dice": float(np.mean([item["dice"] for item in category_metrics.values()])),
                    f"macro_{topology_key}": float(
                        np.mean([item[topology_key] for item in category_metrics.values()])
                    ),
                    "macro_objective": float(
                        np.mean([item["objective"] for item in category_metrics.values()])
                    ),
                    "per_category": category_metrics,
                }
            )
        recommended["contiguous" if contiguous else "isolated"] = max(
            choices, key=lambda item: item["macro_objective"]
        )
    recommended_by_category = {
        category: max(choices, key=lambda item: item["objective"])
        for category, choices in category_choices.items()
    }
    print(
        json.dumps(
            {
                "route_calibration": route_calibration,
                "density_calibration": density_calibration,
                "recommended": recommended,
                "recommended_by_category": recommended_by_category,
                "per_subtype": best,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
