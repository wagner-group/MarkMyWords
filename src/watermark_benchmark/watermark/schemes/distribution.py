import scipy
import torch

from watermark_benchmark.watermark.templates.generator import Watermark
from watermark_benchmark.watermark.templates.verifier import Verifier, EmpiricalVerifier


class DistributionShiftGeneration(Watermark):

    def __init__(self, rng, verifier, tokenizer, temp, delta, gamma):
        super().__init__(rng, verifier, tokenizer, temp)
        self.delta = delta
        self.gamma = gamma
        self.temp  = temp


    def process(self, logits, previous_tokens, ids):

        # Truncate unused logits
        logits = logits[:, :self.rng.vocab_size]

        N, _ = logits.shape
        
        # Get greenlist and update logits
        seeds = self.rng.rand_index(self.rng.get_seed(previous_tokens, ids), 0)
        greenlist = self.rng.green_list(seeds, self.gamma)
        logits[torch.arange(N).unsqueeze(1).expand(-1, greenlist.size(1)), greenlist] += self.delta

        if self.temp > 1e-5:
            logits.div_(self.temp)

        return logits


class DistributionShiftVerifier(Verifier):
    def __init__(self, rng, pvalue, tokenizer, gamma):
        super().__init__(rng, pvalue, tokenizer)
        self.gamma = gamma


    def verify(self, tokens, index=0, exact=False):
        tokens = tokens.squeeze()
        cumul = []
        seen = set()
        for i in range(len(tokens)):
            prev_values = tokens[:i]
            current_token = tokens[i].item()

            seeds = self.rng.rand_index(self.rng.get_seed(prev_values, [index]), 0)
            greenlist = self.rng.green_list(seeds, self.gamma)


            if (current_token, seeds.item()) in seen:
                continue

            seen.add((current_token, seeds.item()))

            if current_token in set(greenlist.squeeze().cpu().numpy()):
                cumul.append(1)
            else:
                cumul.append(0)

        if not len(cumul):
            return [(False, self.gamma, 0.5, 0, 0)]

        rtn = []
        ctr = 0
        for i in range(len(cumul)):
            ctr += cumul[i]
            cnt = i+1
            nd = scipy.stats.binomtest(ctr, cnt, self.gamma, alternative='greater').pvalue
            rtn.append((nd < self.pvalue, ctr/cnt, nd, cnt, i))

        return rtn


class DistributionShiftEmpiricalVerifier(EmpiricalVerifier):

    def __init__(self, rng, pvalue, tokenizer, method, gamma_watermark, gamma_edit_distance):
        super().__init__(rng, pvalue, tokenizer, method, gamma_edit_distance, False)
        self.gamma = gamma_watermark
        self.rand_size=1


    def score_matrix(self, tokens, random_values, index=0):
        _, L, _ = random_values.shape
        random_values = random_values[0, :, 0].reshape(1,L).cpu()

        tokens = tokens.squeeze().to(self.rng.device)
        if not tokens.nelement():
            return None

        greenlists = torch.stack([self.rng.green_list(random_values[:, i], self.gamma, True).squeeze() for i in range(L)])
        greenlists = greenlists.repeat(1 + L//len(tokens), 1)[:len(tokens), :].to(self.rng.device)
        rtn = 1-(greenlists[:,tokens].float())
        return rtn.float()


    def random_score_matrix(self, tokens, random_shape, shared_randomness, index=0):
        """ Produce a random score vector (faster to directly sample the random scores than to sample all random values) """
        _, L, _ = random_shape
        val = torch.cuda.FloatTensor(L,self.rng.vocab_size, device=self.rng.device).uniform_(0,1)[shared_randomness, :].lt(self.gamma)
        return 1-(val[:,tokens.squeeze().to(self.rng.device)].float())
        
        #random_values = torch.rand((1,L), dtype=torch.float32).to(self.rng.device)
        #tokens = tokens.squeeze()
        #greenlists = [set(self.rng.green_list(random_values[:, i%L], self.gamma).squeeze().cpu().numpy()) for i in range(len(tokens))]
        #return torch.tensor([[0 if t.item() in g else 1 for t in tokens] for g in greenlists]).to(self.rng.device)


