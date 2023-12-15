import json
import math
import os
import sys
from dataclasses import replace

import dill
import numpy as np

from watermark_benchmark.utils import get_output_file, load_config
from watermark_benchmark.utils.classes import (
    AggregateResults,
    BenchmarkResults,
    Generation,
    Threshold,
    WatermarkSpec,
)
from watermark_benchmark.utils.summarize import (
    find_threshold,
    ideal_quality_hull,
    summarize_robustness,
    verify_threshold,
)


def run(config_file, generations=None):
    # Load config
    config = (
        load_config(config_file) if type(config_file) == str else config_file
    )

    config.hull_axis = [
        tuple(v) if type(v) == list else v for v in config.hull_axis
    ]
    config.aggregate_thresholds = [
        tuple(v) if type(v) == list else v for v in config.aggregate_thresholds
    ]

    prefix = config.results + ("/validation" if config.validation else "/full")
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    if not config.load_from_save:
        # Load generations`
        generations = (
            Generation.from_file(get_output_file(config))
            if generations is None
            else generations
        )

        attack_dict = {
            str(g.watermark): {} for g in generations if g.watermark is not None
        }
        perturbed_list = set()
        non_watermarked = {}
        for g in generations:
            if not g.efficiency:
                g = replace(g, efficiency=math.inf)
            if not g.watermark:
                if g.temp not in non_watermarked:
                    non_watermarked[g.temp] = []
                non_watermarked[g.temp].append(g)
            else:
                key = str(g.watermark)
                if g.attack not in attack_dict[key]:
                    attack_dict[key][g.attack] = []
                attack_dict[key][g.attack].append(g)
                if g.attack is not None:
                    perturbed_list.add((key, g.id))

        # Compute all attack averages
        summary_attack = {
            k: {
                a: BenchmarkResults(
                    quality=np.mean(
                        [g.rating for g in gs if (k, g.id) in perturbed_list]
                    ),
                    efficiency=np.median(
                        [
                            g.efficiency
                            for g in gs
                            if (k, g.id) in perturbed_list
                        ]
                    ),
                    percent_watermarked=np.mean(
                        [
                            1 if g.efficiency != math.inf else 0
                            for g in gs
                            if (k, g.id) in perturbed_list
                        ]
                    ),
                )
                for a, gs in attack_dict[k].items()
            }
            for k in attack_dict
        }

        summary = {
            k: BenchmarkResults(
                quality=np.mean([g.rating for g in d[None]]),
                efficiency=np.median([g.efficiency for g in d[None]]),
                percent_watermarked=np.mean(
                    [1 if g.efficiency != math.inf else 0 for g in d[None]]
                ),
                robustness=None,
            )
            for k, d in attack_dict.items()
        }

        # Compute baselines
        if config.validation:
            with open(config.results + "/full/baselines.json", "r") as infile:
                baselines = {float(k): v for k, v in json.load(infile).items()}
        else:
            baselines = {
                float(temp): np.mean([g.rating for g in gs])
                for temp, gs in non_watermarked.items()
            }

        # Save to file
        with open(prefix + "/summary_attack.pkl", "wb") as outfile:
            dill.dump(summary_attack, outfile)
        with open(prefix + "/summary.pkl", "wb") as outfile:
            dill.dump(summary, outfile)
        with open(prefix + "/baselines.json", "w") as outfile:
            json.dump(baselines, outfile)
    else:
        with open(prefix + "/summary_attack.pkl", "rb") as infile:
            summary_attack = dill.load(infile)
        with open(prefix + "/baselines.json", "r") as infile:
            baselines = {float(k): v for k, v in json.load(infile).items()}
        with open(prefix + "/summary.pkl", "rb") as outfile:
            summary = dill.load(outfile)

    # Compute robustness
    if config.validation:
        with open(config.results + "/full/robustness_hull.pkl", "rb") as infile:
            robustness_hull = dill.load(infile)
        robustness_summary = summarize_robustness(
            summary_attack,
            threshold=config.threshold,
            existing_hull=robustness_hull,
        )
    else:
        robustness_summary = summarize_robustness(
            summary_attack, threshold=config.threshold
        )

    with open(prefix + "/robustness_hull.pkl", "wb") as outfile:
        dill.dump(robustness_summary, outfile)

    # Summarize all metrics
    summary = {
        k: replace(summary[k], robustness=robustness_summary[k])
        for k in summary
    }

    with open(prefix + "/results.tsv", "w") as outfile:
        outfile.write(BenchmarkResults.to_tsv(summary))

    # Compute aggregate metric
    if config.validation:
        with open(config.results + "/full/thresholds.pkl", "rb") as infile:
            thresholds = dill.load(infile)
    else:
        thresholds = {k: {} for k in config.hull_axis}
    for axis in config.hull_axis:
        agg = {}
        for k in summary:
            ws = WatermarkSpec.from_str(k).to_dict()
            if axis == "rng" or (
                type(axis) != str and any("rng" == v for v in axis)
            ):
                if (
                    ws["rng"] == "Internal"
                    and ws["hash_len"] == 0
                    or ws["rng"] != "Internal"
                    and ws["key_len"] == 1
                ):
                    ws["rng"] = "None"
            if type(axis) == str and axis not in ws:
                continue
            if type(axis) == str and axis == "None":
                key = (ws["temp"],)
            if type(axis) == str:
                key = (ws[axis], ws["temp"])
            else:
                key = (tuple([ws[a] for a in axis]), ws["temp"])
            if key not in agg:
                agg[key] = {}
            agg[key][k] = tuple(
                [
                    round(v, 3)
                    for v in (
                        summary[k].quality,
                        summary[k].efficiency,
                        summary[k].robustness.hull.volume,
                    )
                ]
            )
        if not len(agg.keys()):
            continue

        # Run
        if config.validation:
            for thsd in config.aggregate_thresholds:
                thresholds[axis][thsd] = verify_threshold(
                    agg, thresholds[axis][thsd]
                )
        else:
            for thsd in config.aggregate_thresholds:
                thresholds[axis][thsd] = find_threshold(agg, thsd, baselines)

            # New metric
            optimal_points = {
                w: v.selected
                for w, v in ideal_quality_hull(
                    agg, baselines, [config.max_new_tokens, 0.5]
                ).items()
            }
            with open(prefix + "/optimal_points.tsv", "w") as outfile:
                outfile.write("parameter\ttemp\twatermark")
                for (model, temp), selected in optimal_points.items():
                    for value in selected:
                        outfile.write(
                            "\t{}\t{}\t{}\n".format(model, temp, value)
                        )

    with open(prefix + "/thresholds.pkl", "wb") as outfile:
        dill.dump(thresholds, outfile)

    # Write results to human-readable file
    aggregate_results = [
        AggregateResults(axis, value, temp, Threshold(thsd), q, e, r)
        for axis in thresholds
        for thsd in thresholds[axis]
        for (value, temp), (_, (q, e, r)) in thresholds[axis][thsd].items()
    ]
    with open(prefix + "/aggregate_results.tsv", "w") as outfile:
        outfile.write(AggregateResults.to_tsv(aggregate_results))

    with open(prefix + "/thresholds_results.tsv", "w") as outfile:
        for axis in thresholds:
            for thsd in thresholds[axis]:
                for key, val in thresholds[axis][thsd].items():
                    outfile.write(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                            axis, key, *thsd, val[0], *val[1]
                        )
                    )


def main():
    run(sys.argv[1])


if __name__ == "__main__":
    main()
