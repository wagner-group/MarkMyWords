import multiprocessing
import os
import sys
from dataclasses import replace

multiprocessing.set_start_method("spawn")
from watermark_benchmark.utils import load_config
from watermark_benchmark.utils.classes import Generation, WatermarkSpec


def gen_wrapper(config, watermarks):
    config.baseline = True
    from watermark_benchmark.generate import run as gen_run

    gen_run(config, watermarks)


def detect_wrapper(config, generations):
    from watermark_benchmark.detect import run as detect_run

    detect_run(config, generations)


def perturb_wrapper(config, generations):
    from watermark_benchmark.perturb import run as perturb_run

    perturb_run(config, generations)


def rate_wrapper(config, generations):
    from watermark_benchmark.quality import run as rate_run

    rate_run(config, generations)


def run(
    config, watermarks, GENERATE=True, PERTURB=True, RATE=True, DETECT=True
):
    # Generation
    generations = []

    # Create output dir:
    try:
        os.mkdir(config.results)
    except Exception:
        pass

    print("### GENERATING ###")

    if GENERATE:
        config.input_file = None
        config.output_file = config.results + "/generations.tsv"
        gen_wrapper(config, watermarks)

    print("### PERTURBING ###")

    # Perturb
    if PERTURB:
        config.input_file = config.results + "/generations.tsv"
        config.output_file = config.results + "/perturbed.tsv"
        generations = Generation.from_file(config.input_file)
        perturb_wrapper(config, generations)

    print("### RATING ###")

    # Rate
    if RATE:
        config.input_file = config.results + "/perturbed.tsv"
        config.output_file = config.results + "/rated.tsv"
        generations = Generation.from_file(config.input_file)
        rate_wrapper(config, generations)

    print("### DETECTING ###")

    # Detect
    if DETECT:
        config.input_file = config.results + "/rated.tsv"
        config.output_file = config.results + "/detect.tsv"
        generations = Generation.from_file(config.input_file)
        detect_wrapper(config, generations)
        generations = Generation.from_file(config.output_file)

    return generations


def main():
    config = load_config(sys.argv[1])
    config.validation = False
    with open(config.watermark) as infile:
        watermarks = [
            replace(WatermarkSpec.from_str(l.strip()), tokenizer=config.model)
            for l in infile.read().split("\n")
            if len(l)
        ]
    generations = run(config, watermarks)

    # Summary
    from watermark_benchmark.summarize import run as summary_run

    summary_run(config, generations)

    return

    # config.validation = True
    # config.output += '_validation'
    # config.done_tasks += '_validation'
    # validation_generations = run(config, watermarks)

    # summary_run(config, validation_generations)
