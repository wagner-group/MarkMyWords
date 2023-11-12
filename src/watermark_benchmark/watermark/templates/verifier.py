from abc  import ABC, abstractmethod
import hash_cpp
import torch
import numpy as np

from watermark_benchmark.watermark.templates.random import ExternalRandomness

from abc import ABC, abstractmethod

class Verifier(ABC):
    """
    Abstract base class for watermark verifiers.
    """

    @abstractmethod
    def __init__(self, rng, pvalue, tokenizer):
        """
        Initializes the Verifier object.

        :param rng: The random number generator to use.
        :type rng: Random
        :param pvalue: The p-value to use for the verification.
        :type pvalue: float
        :param tokenizer: The tokenizer to use for splitting text into tokens.
        :type tokenizer: Tokenizer
        """
        self.pvalue = pvalue
        self.rng = rng
        self.tokenizer = tokenizer


    @abstractmethod
    def verify(self, tokens, index=0, exact=False):
        """
        Verifies if a given sequence of tokens contains a watermark.

        :param tokens: The sequence of tokens to verify.
        :type tokens: list of str
        :param index: The starting index to search for the watermark.
        :type index: int
        :param exact: Whether to search for an exact match of the watermark or a partial match.
        :type exact: bool
        :return: True if the watermark is found, False otherwise.
        :rtype: bool
        """
        pass

    def id(self):
        """
        Returns a unique identifier for the verifier.

        :return: A tuple containing the p-value, the type of the verifier, and the version of the verifier.
        :rtype: tuple
        """
        return (self.pvalue, "theoretical", "standard")



class EmpiricalVerifier(Verifier):
    """
    EmpiricalVerifier is a subclass of Verifier that implements the empirical verification method.
    It uses a score matrix to detect watermarks in a given text.
    """
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
        """
        Detects the watermark in the given score matrix using the specified method.

        Args:
            scores (torch.Tensor): The score matrix.

        Returns:
            torch.Tensor: The detected watermark.
        """
        if self.method == "regular":
            A = self.regular_distance(scores)
        else:
            A = self.levenstein_distance(scores)
        return A.min(axis=0)


    def regular_distance(self, scores):
        """
        Computes the regular distance between the scores.

        Args:
            scores (torch.Tensor): The score matrix.

        Returns:
            torch.Tensor: The computed distance.
        """
        KL, SL = scores.shape
        if type(self.rng) == ExternalRandomness:
            indices = torch.vstack((((torch.arange(KL).reshape(-1,1).repeat(1,SL).flatten() + torch.arange(SL).repeat(KL)) % KL), torch.arange(SL).repeat(KL)%SL)).t().reshape(KL, -1, 2).to(scores.device)
            rslt = scores[indices[:,:,0], indices[:,:,1]].cumsum(axis=1)
        else:
            rslt = scores[torch.arange(SL), torch.arange(SL)].cumsum(0).unsqueeze(0)
        return rslt


    def levenstein_distance(self, scores):
        """
        Computes the Levenstein distance between the scores.

        Args:
            scores (torch.Tensor): The score matrix.

        Returns:
            torch.Tensor: The computed distance.
        """
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
        """
        Pre-computes a set of baseline scores to speed up the verification process.

        Args:
            max_len (int, optional): The maximum length of the text. Defaults to 1024.
            runs (int, optional): The number of runs to perform. Defaults to 200.
        """
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
        """
        Verifies if the given text contains a watermark.

        Args:
            tokens (torch.Tensor): The text to verify.
            index (int, optional): The index of the text. Defaults to 0.
            exact (bool, optional): Whether to perform an exact verification. Defaults to False.

        Returns:
            list: A list of tuples containing the verification results.
        """
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
        """
        Returns the ID of the verifier.

        Returns:
            tuple: The ID of the verifier.
        """
        return (self.pvalue, "empirical", self.method)

