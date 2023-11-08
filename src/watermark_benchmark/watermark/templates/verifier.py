from abc  import ABC, abstractmethod
import hash_cpp
import torch
import numpy as np

from watermark_benchmark.watermark.templates.random import ExternalRandomness

class Verifier(ABC):

    @abstractmethod
    def __init__(self, rng, pvalue, tokenizer):
        self.pvalue = pvalue
        self.rng = rng
        self.tokenizer = tokenizer


    @abstractmethod
    def verify(self, tokens, index=0, exact=False):
        pass

    def id(self):
        return (self.pvalue, "theoretical", "standard")



class EmpiricalVerifier(Verifier):

    @abstractmethod
    def __init__(self, rng, pvalue, tokenizer, method, gamma, log):
        super().__init__(rng, pvalue, tokenizer)
        self.method = method
        self.precomputed = False
        self.gamma_edit = gamma if not log else np.log(gamma)
        self.rand_size = self.rng.vocab_size


    @abstractmethod
    def score_matrix(self, tokens, random_values, index=0):
        pass


    @abstractmethod
    def random_score_matrix(self, tokens, random_shape, shared_randomness, index=0):
        pass


    def detect(self, scores):
        if self.method == "regular":
            A = self.regular_distance(scores)
        else:
            A = self.levenstein_distance(scores)
        return A.min(axis=0)


    def regular_distance(self, scores):
        KL, SL = scores.shape
        if type(self.rng) == ExternalRandomness:
            indices = torch.vstack((((torch.arange(KL).reshape(-1,1).repeat(1,SL).flatten() + torch.arange(SL).repeat(KL)) % KL), torch.arange(SL).repeat(KL)%SL)).t().reshape(KL, -1, 2).to(scores.device)
            rslt = scores[indices[:,:,0], indices[:,:,1]].cumsum(axis=1)
        else:
            rslt = scores[torch.arange(SL), torch.arange(SL)].cumsum(0).unsqueeze(0)
        return rslt


    def levenstein_distance(self, scores):
        KL, SL = scores.shape
        if type(self.rng) == ExternalRandomness:
            container = torch.zeros((KL, SL+1, SL+1)).float().to(self.rng.device) 
        else:
            container = torch.zeros((1, SL+1, SL+1)).float().to(self.rng.device)

        # Set initial values
        container[:, 0, :] = (torch.arange(SL+1)*self.gamma_edit).to(self.rng.device).unsqueeze(0).expand(container.shape[0], -1)
        container[:, :, 0] = (torch.arange(SL+1)*self.gamma_edit).to(self.rng.device).unsqueeze(0).expand(container.shape[0], -1)

        # Compute
        container = hash_cpp.levenshtein(scores, container, self.gamma_edit)
        return container[:,torch.arange(1,SL+1),torch.arange(1,SL+1)]


    def pre_compute_baseline(self, max_len=1024, runs=200):
        # To speed things up, we pre-compute a set of baseline scores. This should work well for external randomness, for internal randomness it can induce FPs since repeated tokens can induce repeated randomness
        self.precomputed_results = torch.zeros((runs, max_len)).to(self.rng.device)
        tokens = torch.randint(0,self.rng.vocab_size,(max_len,))
        if type(self.rng) == ExternalRandomness: 
            shared_randomness = torch.arange(self.rng.key_len).repeat(1 + max_len // self.rng.key_len)[:max_len].to(self.rng.device)
            L = self.rng.key_len
        else:
            shared_randomness = torch.arange(max_len).to(self.rng.device)
            L = max_len

        for run in range(runs):
            scores = self.random_score_matrix(tokens, (1, L, self.rng.vocab_size), shared_randomness)
            self.precomputed_results[run] = self.detect(scores)[0]

        self.precomputed = True



    def verify(self, tokens, index=0, exact=False):
        tokens = tokens.to(self.rng.device)

        if type(self.rng) == ExternalRandomness:
            xi = self.rng.xi[index].to(self.rng.device).unsqueeze(0)
            scores = self.score_matrix(tokens, xi, index=index)
        else:
            if self.rand_size > 1:
                randomness = torch.cat(tuple(self.rng.rand_range(self.rng.get_seed(tokens[:,:i], [index]), self.rand_size) for i in range(tokens.shape[-1])), axis=0).unsqueeze(0)
            else:
                randomness = torch.cat(tuple(self.rng.rand_index(self.rng.get_seed(tokens[:,:i], [index]), 0).reshape(1,1) for i in range(tokens.shape[-1])), axis=0).unsqueeze(0)

            xi = randomness
            scores = self.score_matrix(tokens, randomness, index=index)

        if scores is None:
            return [(False, 0, 0, 0)]

        test_result = self.detect(scores)[0]
        p_val = torch.zeros_like(test_result).to(self.rng.device)

        if exact:
            rc = 100
            # Before simlating random seeds, we need to figure out which tokens will share the same randomness
            if type(self.rng) == ExternalRandomness:
                shared_randomness = torch.arange(self.rng.key_len)
            else:
                _, shared_randomness = xi[0,:,0].unique(return_inverse=True)
                shared_randomness = shared_randomness.to(self.rng.device)


            #rv = torch.cuda.FloatTensor(100, xi.shape[1], tokens.shape[-1]).uniform_(0,1).to(self.rng.device)
            for _ in range(rc):
                scores_alt = self.random_score_matrix(tokens, xi.shape, shared_randomness, index=index)
                null_result = self.detect(scores_alt)[0]
                p_val += (null_result < test_result)

        else:
            rc = 100
            if not self.precomputed:
                self.pre_compute_baseline()
            null_result = self.precomputed_results[torch.randperm(self.precomputed_results.shape[0])[:100].to(self.rng.device), :test_result.shape[-1]]
            if null_result.shape[-1] < test_result.shape[-1]:
                test_result = test_result[:null_result.shape[-1]]
            p_val = (null_result < test_result).sum(axis=0) 


        rtn = []
        for idx, val in enumerate(p_val.cpu().numpy()):
            rtn.append((val/rc <= self.pvalue, val/rc, val/rc, idx))

        self.rng.reset()
        return rtn

    def id(self):
        return (self.pvalue, "empirical", self.method)

