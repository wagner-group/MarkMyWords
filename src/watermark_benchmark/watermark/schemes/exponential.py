import random
import math
import scipy
import torch
import torch.nn.functional as F
import numpy as np

from watermark_benchmark.watermark.templates.generator import Watermark
from watermark_benchmark.watermark.templates.verifier import Verifier, EmpiricalVerifier


class ExponentialGenerator(Watermark):

    def __init__(self, rng, verifiers, tokenizer, temp, skip_prob): 
        super().__init__(rng, verifiers, tokenizer, temp)
        self.skip_prob = skip_prob

    def process(self, logits, previous_tokens, ids):

        #print("Previous tokens: {}".format([p[-1] for p in previous_tokens]))
        #print(id(self))

        # Truncate unused logits. Return as is with skip probability
        logits = logits[:, :self.rng.vocab_size] / self.temp
        if random.random() < self.skip_prob:
            return logits

        # Compute probabilities and get random values
        probs = F.softmax(logits, dim=-1)
        hash_values = self.rng.rand_range(self.rng.get_seed(previous_tokens, ids), self.rng.vocab_size, probs.device)
        hash_values = torch.div(-torch.log(hash_values), probs)

        # Get next token, and update logit
        next_token = hash_values.argmin(dim=-1)

        #print("Next token choice: {} | {} (P = {})".format(next_token.cpu(), hash_values.min(dim=-1).cpu(), probs[next_token.to(probs.device)].cpu()))

        logits[:] = -math.inf
        logits[torch.arange(logits.shape[0]), next_token] = 0

        #print("Next tokens: {}".format(next_token))

        return logits


class ExponentialVerifier(Verifier):

    def __init__(self, rng, pvalue, tokenizer, log):
        super().__init__(rng, pvalue, tokenizer)
        self.log = log

    def verify(self, tokens, index=0, exact=False):
        tokens = tokens.squeeze()
        cumul = []
        seen = set()

        if not tokens.numel():
            return [(False, 0.5, 0.5, 0, 0)]    
        try:
            for i, tok in enumerate(tokens):
                prev_values = tokens[:i]
                seed = self.rng.get_seed(prev_values, [index])
                hv = self.rng.rand_index(seed, tok).item()

                if (seed[0], hv) in seen:
                    continue

                seen.add((seed[0], hv))
                cumul.append((hv,i))
        except Exception:
            cumul=[]

        if not len(cumul):
            return [(False, 0.5, 0.5, 0, 0)]

        rtn = []
        ctr = 0
        ctn = 0
        for i, val in enumerate(cumul):
            ctn += 1
            ctr += val[0] if not self.log else -np.log(max(0.00001, 1-val[0]))
            if not self.log:
                #pval = tfp.distributions.Bates(ctn).survival_function(ctr/ctn)
                pval = scipy.stats.norm.sf(ctr/ctn, loc=0.5, scale=1/math.sqrt(12*ctn))
            else:
                #pval = s(ctr, loc=0.5, scale=1/math.sqrt(12*(ctn))) if not self.log else scipy.stats.gamma.sf(ctr, ctn)
                pval = scipy.stats.gamma.sf(ctr, ctn)
            rtn.append((pval < self.pvalue, ctr/ctn if not self.log else ctr, pval, i+1))
        return rtn

    def id(self):
        base = super().id()
        return base[:2] + ("log",) if self.log else base


class ExponentialEmpiricalVerifier(EmpiricalVerifier):

    def __init__(self, rng, pvalue, tokenizer, method, log, gamma):
        log = True
        super().__init__(rng, pvalue, tokenizer, method, gamma, log=log)
        self.log = log

    def score_matrix(self, tokens, random_values, index=0):
        """ Prepare all possible overlapping of random values (shape KEY_LEN x VOCAB_SIZE) and tokens (shape SEQ_LEN) """
        tokens = tokens.squeeze()
        if not tokens.nelement():
            return None
        random_values = random_values.squeeze()
        return torch.log(1-random_values[:,tokens]) if self.log else 1-random_values[:,tokens]

    def random_score_matrix(self, tokens, random_shape, shared_randomness, index=0):
        """ Produce a random score vector (faster to directly sample the random scores than to sample all random values) """
        # To do: repeat random values when token context is the same
        _, L, V = random_shape
        tokens = tokens.squeeze()
        if not tokens.nelement():
            return None
        random_values = torch.cuda.FloatTensor(L, V, device=self.rng.device).uniform_(0,1)[shared_randomness,:]
        return torch.log(1-random_values[:,tokens]) if self.log else 1-random_values[:,tokens]

    def id(self):
        base = super().id()
        return base[:2] + ("log",) if self.log else base


