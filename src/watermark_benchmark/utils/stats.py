""" Generation entropy statistics """
import torch

class Stats:
    """ Object to compute statistics about generation entropy """

    def __init__(self, l, t):
        self.se = torch.tensor([0.0 for _ in range(l)]).float()
        self.e  = torch.tensor([0.0 for _ in range(l)]).float()
        self.c  = torch.tensor([0 for _ in range(l)]).long()
        self.t  = t


    def update(self, logits, ids):
        """ To run for every token sampling in order to update entropy """ 
        ids = [int(i) for i in ids]
        # t = torch.tensor([self.t for _ in range(logits.shape[0])], \
        #        dtype=logits.dtype, \
        #        device=logits.device)
        #logits = logits.div(t.unsqueeze(dim=1))
        probs  = torch.softmax(logits, dim=-1, dtype=torch.float)
        logprobs = torch.log(probs)
        logprobs[probs < 1e-5] = 0

        updated_indices = torch.tensor(ids)
        self.se[updated_indices] += (probs / (1 + probs)).sum(axis=-1).cpu()
        self.e[updated_indices]  += -(probs*logprobs).sum(axis=-1).cpu()
        self.c[updated_indices]  += 1

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices, )

        rtn = []
        for index in indices:
            c  = self.c[index].item()
            if not c:
                se, e = 0, 0
            else:
                se = self.se[index].item() / c
                e  = self.e[index].item()  / c
            rtn.append((c,e,se))

        if len(rtn) == 1:
            return rtn[0]
        else:
            return rtn
