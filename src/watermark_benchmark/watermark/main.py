import yaml  # type: ignore
import torch
import torch.nn.functional as F

from watermark_benchmark.utils.classes import WatermarkSpec
from watermark_benchmark.watermark.templates.generator import Watermark
from watermark_benchmark.watermark.templates.random import EmbeddedRandomness, ExternalRandomness
from watermark_benchmark.watermark.schemes.distribution import DistributionShiftEmpiricalVerifier, DistributionShiftGeneration, DistributionShiftVerifier
from watermark_benchmark.watermark.schemes.its import InverseTransformEmpiricalVerifier, InverseTransformVerifier, InverseTransformGenerator
from watermark_benchmark.watermark.schemes.exponential import ExponentialEmpiricalVerifier, ExponentialGenerator, ExponentialVerifier
from watermark_benchmark.watermark.schemes.binary import BinaryEmpiricalVerifier, BinaryGenerator, BinaryVerifier

class NoWatermark(Watermark):

    def __init__(self):
        super().__init__(None, None, None, None)

    def next_token_select(self, logits, previous_tokens):
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_tokens

    def verify(self, tokens, index=0, exact=False, skip_edit=False):
        return [(False, 0.5, 0.5, 0)]

    def verify_text(self, text, index=0, exact=False, skip_edit=False):
        return [(False, 0.5, 0.5, 0)]



### EXPORT ###

def get_watermark_spec(schema_path: str) -> WatermarkSpec:
    with open(schema_path, "r") as f:
        raw = yaml.safe_load(f)
    return WatermarkSpec.from_dict(raw)


class WatermarkNotFoundError(Exception):
    pass


def get_watermark(watermark_spec, tokenizer, binarizer=None, device="cpu", key=None):
    # Parse parameters
    if not tokenizer:
        tokenizer = watermark_spec.tokenizer_engine

    key = watermark_spec.secret_key if key is None else key

    # Randomness
    rng = None
    if watermark_spec.rng == "Internal":
        rng = EmbeddedRandomness(key, device, len(tokenizer), watermark_spec.hash_len, watermark_spec.min_hash)
    else:
        if watermark_spec.generator == "distributionshift" or watermark_spec.generator == "its":
            rng = ExternalRandomness(key, device, len(tokenizer), watermark_spec.key_len, 1)
        elif watermark_spec.generator == "binary":
            rng = ExternalRandomness(key, device, len(tokenizer), watermark_spec.key_len, binarizer.L)
        else:
            rng = ExternalRandomness(key, device, len(tokenizer), watermark_spec.key_len)

    # Verifier
    verifiers = []
    for v_spec in watermark_spec.verifiers:
        if v_spec.verifier == "Empirical":
            if watermark_spec.generator == "distributionshift":
                verifier = DistributionShiftEmpiricalVerifier(rng,watermark_spec.pvalue, tokenizer, v_spec.empirical_method, watermark_spec.gamma, v_spec.gamma)
            elif watermark_spec.generator == "exponential":
                verifier = ExponentialEmpiricalVerifier(rng,watermark_spec.pvalue, tokenizer, v_spec.empirical_method,v_spec.log, v_spec.gamma)
            elif watermark_spec.generator == "its":
                verifier = InverseTransformEmpiricalVerifier(rng, watermark_spec.pvalue, tokenizer, v_spec.empirical_method, v_spec.gamma) 
            elif watermark_spec.generator == "binary":
                verifier = BinaryEmpiricalVerifier(rng, watermark_spec.pvalue, tokenizer, v_spec.empirical_method, binarizer, watermark_spec.skip_prob, v_spec.gamma)
            else:
                raise WatermarkNotFoundError
        else:
            if watermark_spec.generator == "distributionshift":
                verifier = DistributionShiftVerifier(rng, watermark_spec.pvalue, tokenizer, watermark_spec.gamma)
            elif watermark_spec.generator == "exponential":
                verifier = ExponentialVerifier(rng,  watermark_spec.pvalue, tokenizer, v_spec.log)
            elif watermark_spec.generator == "its":
                verifier = InverseTransformVerifier(rng, watermark_spec.pvalue, tokenizer)
            elif watermark_spec.generator == "binary":
                verifier = BinaryVerifier(rng, watermark_spec.pvalue, tokenizer, binarizer, watermark_spec.skip_prob)
            else:
                raise WatermarkNotFoundError
        verifiers.append(verifier)

    # Generator
    if watermark_spec.generator == "distributionshift":
        return DistributionShiftGeneration(rng, verifiers, tokenizer, watermark_spec.temp, watermark_spec.delta, watermark_spec.gamma)
    elif watermark_spec.generator == "exponential":
        return ExponentialGenerator(rng,verifiers, tokenizer, watermark_spec.temp, watermark_spec.skip_prob)
    elif watermark_spec.generator == "its":
        return InverseTransformGenerator(rng, verifiers, tokenizer, watermark_spec.temp, watermark_spec.skip_prob)
    elif watermark_spec.generator == "binary":
        return BinaryGenerator(rng, verifiers, tokenizer, watermark_spec.temp, binarizer, watermark_spec.skip_prob)
    else:
        raise WatermarkNotFoundError



