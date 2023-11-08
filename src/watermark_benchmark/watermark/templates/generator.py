from abc import ABC, abstractmethod

class Watermark(ABC):
    
    @abstractmethod
    def __init__(self, rng, verifiers, tokenizer, temp):
        self.rng = rng
        self.verifiers = verifiers
        self.tokenizer = tokenizer
        self.temp = temp

    @abstractmethod
    def process(self, logits, previous_tokens, ids):
        logits.div_(self.temp)
        return logits

    def reset(self):
        self.rng.reset()

    def verify(self, tokens, index=0, exact=False, skip_edit=False):
        rtn = []
        for v in self.verifiers:
            if 'method' in v.__dict__ and v.method != "regular" and skip_edit:
                continue
            if 'method' in v.__dict__ and v.method != "regular":
                # Don't use exact for edit distance since it's too slow
                exact = False
            rtn.append((v.id(), v.verify(tokens, index=index, exact=exact)))
        return rtn

    def verify_text(self, text, index=0, exact=False, skip_edit=False):
        tokens = self.tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').to(self.rng.device)
        return self.verify(tokens, index=index, exact=exact, skip_edit=skip_edit)

