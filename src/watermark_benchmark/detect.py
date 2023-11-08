import os
import sys
from tqdm import tqdm
from dataclasses import replace
import math
import random
import multiprocessing
import signal


def writer_process(queue, config, w_count):
    from watermark_benchmark.utils import get_output_file
    outfilepath = get_output_file(config)
    detailed = outfilepath.replace(".tsv", "_detailed.tsv")

    for _ in tqdm(range(w_count), total=w_count, desc="Detect"):
        task = queue.get(block=True)
        if task is None:
            queue.put(None)
            return

        full, reduced = task

        with open(outfilepath, "a") as outfile:
            outfile.write("\n".join(str(gen) for gen in full)+"\n")

        if len(reduced):
            with open(detailed, "a") as outfile:
                outfile.write("\n".join(str(gen) for gen in reduced)+"\n")


def get_efficiency(w, pval):
    if w[-1][2] > pval:
        return math.inf
    
    idx = len(w) - 1
    while idx >= 0 and w[idx][2] <= pval:
        idx -= 1

    return w[idx+1][3]


def detect_process(device, config, tasks, writer_queue):

    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    import torch
    from watermark_benchmark.utils.bit_tokenizer import Binarization
    from watermark_benchmark.utils import setup_randomness, get_tokenizer
    from watermark_benchmark.watermark import get_watermark

    torch.set_num_threads(1)

    # Randomness and models
    tokenizer = get_tokenizer(config.model)
    binarizer = Binarization(tokenizer, [0], use_huffman_coding=config.huffman_coding is not None, huffman_coding_path=config.huffman_coding)

    for task in tasks:
        setup_randomness(config)
        _, generations = task
        watermark = generations[0].watermark
        reduced, full = [], []
        unique_keys, keys, key_indices = {}, [], []
        for g in generations:
            if g.key not in unique_keys:
                unique_keys[g.key] = len(keys)
                keys.append(g.key)
            key_indices.append(unique_keys[g.key])

        try:
            watermark_engine = get_watermark(watermark, tokenizer, binarizer, [0], keys)
            for g_idx, g in enumerate(generations):
                W = watermark_engine.verify_text(g.response, exact=True, index=key_indices[g_idx])
                sep_watermarks = watermark.sep_verifiers()

                for w_i, w in enumerate(W):
                    pvalue = w[1][-1][2]
                    eff = get_efficiency(w[1], watermark.pvalue)
                    full.append(replace(g, watermark=replace(sep_watermarks[w_i], secret_key=g.key), pvalue=pvalue, efficiency=eff))
                    reduced.extend([replace(g, response="", token_count = v[3], pvalue = v[2], watermark=replace(sep_watermarks[w_i], secret_key=g.key)) for v in w[1]])

            writer_queue.put((full, reduced))

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"\n\n\n##### OOM for {watermark} #####\n\n\n\n")
                continue
            else:
                print("\n\n\n##### RUNTIME ERROR #####\n\n\n\n")
                raise e

        except Exception as e:
            print("\n\n\n##### ERROR #####\n\n\n\n")
            raise e


def run(config, generations=None):

    from watermark_benchmark.utils.classes import Generation
    from watermark_benchmark.utils import load_config, setup_randomness, get_output_file, get_input_file

    # load config
    config = load_config(config) if type(config) == str else config
    setup_randomness(config)

    # load generations
    generations = Generation.from_file(get_input_file(config)) if not generations else generations
    outfilepath = get_output_file(config)
    if not os.path.exists(outfilepath):
        Generation.to_file(outfilepath)
    existing = {str(g.watermark.to_dict(True, True) if g.watermark is not None else g.temp) for g in Generation.from_file(outfilepath)}
    specs = {str(g.watermark.to_dict(True, True)): [] for g in generations if g.watermark is not None and str(g.watermark.to_dict(True, True)) not in existing}
    baselines = {str(g.temp): [] for g in generations if g.watermark is None and str(g.temp) not in existing}

    tl = len(specs) + len(baselines)
    if not tl:
        return

    for g in generations:
        if g.watermark is None and str(g.temp) in baselines:
            baselines[str(g.temp)].append(g)
        elif g.watermark is not None and str(g.watermark.to_dict(True, True)) in specs:
            specs[str(g.watermark.to_dict(True, True))].append(g)

    # Setup writer
    global_manager = multiprocessing.Manager()
    writer_queue = global_manager.Queue()
    writer = multiprocessing.Process(target=writer_process, args=(writer_queue, config, len(specs) + len(baselines) ))
    writer.start()

    for t in baselines:
        writer_queue.put((baselines[t], []))
    
    if len(specs):
        devices = config.get_devices()
        task_count = 4*len(devices)
        all_tasks = list(specs.items())
        random.shuffle(all_tasks)
        tasks_per_process = [[] for _ in range(task_count)]

        for t_idx, (w, g) in enumerate(all_tasks):
            tasks_per_process[t_idx%task_count].append((w,g))

        detect_processes = []
        for t_idx, t in enumerate(tasks_per_process):
            detect_processes.append(multiprocessing.Process(target=detect_process, args=(devices[t_idx % len(devices)], config, t, writer_queue)))
            detect_processes[-1].start()

    # Setup signal handler
    def graceful_exit(sig, frame):
        print("Stopping all processes...")
        for p in detect_processes:
            p.terminate()
        writer.terminate()
        exit()
    
    signal.signal(signal.SIGINT, graceful_exit)

    writer.join()
    for p in detect_processes:
        p.terminate()



def main():
    multiprocessing.set_start_method('spawn')
    run(sys.argv[1])

