""" Generation entropy statistics """
import torch


class Stats:
    """Object to compute statistics about generation entropy"""

    def __init__(self, l, t, logprobs=False):
        self.se = torch.tensor([0.0 for _ in range(l)]).float()
        self.e = torch.tensor([0.0 for _ in range(l)]).float()
        self.c = torch.tensor([0 for _ in range(l)]).long()
        self.t = t
        self.save_logprobs = logprobs
        self.logprobs = [[] for _ in range(l)]

    def update(self, logits, ids):
        """To run for every token sampling in order to update entropy"""
        ids = [int(i) for i in ids]
        # t = torch.tensor([self.t for _ in range(logits.shape[0])], \
        #        dtype=logits.dtype, \
        #        device=logits.device)
        # logits = logits.div(t.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        logprobs = torch.log(probs)
        logprobs[probs < 1e-5] = 0

        updated_indices = torch.tensor(ids)
        self.se[updated_indices] += (probs / (1 + probs)).sum(axis=-1).cpu()
        self.e[updated_indices] += -(probs * logprobs).sum(axis=-1).cpu()
        self.c[updated_indices] += 1

        if self.save_logprobs:
            for tensor_id, id in enumerate(ids):
                local_logprobs = logprobs[tensor_id]
                if isinstance(local_logprobs, torch.Tensor):
                    local_logprobs = local_logprobs.squeeze().cpu().numpy()
                d = {i: v for i, v in enumerate(local_logprobs)}
                self.logprobs[id].append(d)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)

        rtn = []
        for index in indices:
            c = self.c[index].item()
            if not c:
                se, e = 0, 0
            else:
                se = self.se[index].item() / c
                e = self.e[index].item() / c
            rtn.append((c, e, se))

        if len(rtn) == 1:
            return rtn[0]
        else:
            return rtn
