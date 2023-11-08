import math
import random
import torch
import torch.nn.functional as F
import scipy

from watermark_benchmark.watermark.templates.generator import Watermark
from watermark_benchmark.watermark.templates.verifier import Verifier, EmpiricalVerifier
from watermark_benchmark.watermark.templates.random import ExternalRandomness

class InverseTransformGenerator(Watermark):

    def __init__(self, rng, verifiers, tokenizer, temp, skip_prob):
        super().__init__(rng, verifiers, tokenizer, temp)
        self.skip_prob = skip_prob
        self.ctr = None


    def process(self, logits, previous_tokens, ids):
        # Truncate unused logits
        logits = logits[:, :self.rng.vocab_size] / self.temp

        if random.random() < self.skip_prob:
            return logits

        # Use float32 to avoid floating point errors leading to the sum of probabilities being less than 1
        probs = F.softmax(logits.float(), dim=-1)

        # Get random values
        u = self.rng.rand_index(self.rng.get_seed(previous_tokens, ids), 0, device=logits.device)

        permutation = self.rng.get_permutation(logits.device)
        ids = torch.Tensor(ids).long().to(probs.device)

        # Convert to float to avoid floating point errors
        probs = probs.float()
        selected = torch.searchsorted(probs[torch.arange(probs.shape[0]).unsqueeze(1).to(probs.device), permutation[ids, :]].cumsum(axis=1), u.unsqueeze(0).t(), side="right").squeeze()
        selected[selected == self.rng.vocab_size] -= 1
        next_token = permutation[ids, selected]
        
        logits = torch.full(logits.shape, -math.inf).to(logits.device)
        logits[torch.arange(logits.shape[0]), next_token] = 0

        return logits



class InverseTransformVerifier(Verifier):

    def __init__(self, rng, pvalue, tokenizer):
        super().__init__(rng, pvalue, tokenizer)
        self.log = False

    def verify(self, tokens, index=0, exact=False):

        tokens = tokens.to(self.rng.device).squeeze()
        if type(self.rng) == ExternalRandomness:
            xi = self.rng.xi[index].to(self.rng.device)[:,0]
        else:
            xi = torch.cat(tuple(self.rng.rand_range(self.rng.get_seed(tokens[:i], [index]), 1) for i in range(tokens.shape[-1])), axis=0)[:,0]
        inv_permutation = self.rng.get_permutation(tokens.device, True)
        scores = (xi.repeat(1 + tokens.shape[0] // xi.shape[-1])[:tokens.shape[0]] - (inv_permutation[index, tokens].float()/(self.rng.vocab_size-1))).abs().contiguous().cumsum(0)

        if not len(scores):
            return [(False, 0.5, 0.5, 0, 0)]

        rtn = []
        for i, val in enumerate(scores.tolist()):
            pval = scipy.stats.norm.cdf(val/(i+1), loc=1/3, scale=2/(math.sqrt(18*(i+1))))
            rtn.append((pval < self.pvalue, val/(i+1), pval, i+1))

        return rtn


class InverseTransformEmpiricalVerifier(EmpiricalVerifier):

    def __init__(self, rng, pvalue, tokenizer, method, gamma):
        super().__init__(rng, pvalue, tokenizer, method, gamma, log=False)
        self.rand_size=1


    def score_matrix(self, tokens, random_values, index=0):
        """ Prepare all possible overlapping of random values (shape KEY_LEN) and tokens (shape SEQ_LEN) """
        random_values = random_values[0,:,0]
        #torch.rand(random_values[0,:,0].shape).to(random_values.device)
        tokens = tokens.squeeze()
        if not tokens.nelement():
            return None
        inv_permutation = self.rng.get_permutation(tokens.device, True)
        rtn = (random_values.repeat(tokens.shape[0],1).t() - (inv_permutation[index, tokens].float().repeat(random_values.shape[0],1)/(self.rng.vocab_size-1))).abs().contiguous()
        return rtn


    def random_score_matrix(self, tokens, random_shape, shared_randomness, index=0):
        """ Produce a random score vector (faster to directly sample the random scores than to sample all random values) """
        _, L, _ = random_shape
        random_values = torch.cuda.FloatTensor(L, device=self.rng.device).uniform_(0,1)[shared_randomness]
        tokens = tokens.squeeze().to(self.rng.device)
        if not tokens.nelement():
            return None
        inv_permutation = self.rng.get_permutation(tokens.device, True)
        return (random_values.repeat(tokens.shape[0],1).t() - (inv_permutation[index, tokens].float().repeat(random_values.shape[0],1)/(self.rng.vocab_size-1))).abs().contiguous()
