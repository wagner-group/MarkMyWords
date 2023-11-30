# utils folder

import random

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from watermark_benchmark.utils.classes import ConfigSpec


def load_config(config_file):
    # Load arguments
    with open(config_file, "r") as infile:
        config = yaml.safe_load(infile)
    return ConfigSpec.from_dict(config)


def setup_randomness(config):
    # Setup randomness
    if config.seed is not None:
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)


def get_server_args(config):
    # Setup server
    additional_keywords = [
        "gpu_memory_utilization",
        "seed",
        "dtype",
        "trust_remote_code",
    ]
    additional_args = {
        key: config.__dict__[key]
        for key in additional_keywords
        if config.__dict__[key] is not None
    }
    return additional_args


def get_output_file(config):
    # Setup outfile
    return config.output_file


def get_input_file(config):
    # Setup outfile
    return config.input_file


def get_tokenizer(model):
    # Return HF tokenizer
    return AutoTokenizer.from_pretrained(model)
