from abc import ABC, abstractmethod


class Watermark(ABC):
    """
    Abstract base class for watermarking algorithms.

    Attributes:
        rng: An instance of the random number generator.
        verifiers: A list of verification algorithms.
        tokenizer: An instance of the tokenizer.
        temp: A temperature parameter used in the watermarking process.
    """

    @abstractmethod
    def __init__(self, rng, verifiers, tokenizer, temp):
        self.rng = rng
        self.verifiers = verifiers
        self.tokenizer = tokenizer
        self.temp = temp

    @abstractmethod
    def process(self, logits, previous_tokens, ids):
        """
        Abstract method for processing logits.

        Args:
            logits: A tensor of logits.
            previous_tokens: A tensor of previous tokens.
            ids: A tensor of ids.

        Returns:
            A tensor of processed logits.
        """
        # logits.div_(self.temp)
        return logits

    def reset(self):
        """
        Resets the random number generator.
        """
        self.rng.reset()

    def verify(self, tokens, index=0, exact=False, skip_edit=False):
        """
        Verifies the watermark in the given tokens.

        Args:
            tokens: A tensor of tokens.
            index: The index of the token to verify.
            exact: Whether to use exact matching.
            skip_edit: Whether to skip edit distance verification.

        Returns:
            A list of tuples containing the verifier ID and the verification result.
        """
        rtn = []
        for v in self.verifiers:
            if "method" in v.__dict__ and v.method != "regular" and skip_edit:
                continue
            if "method" in v.__dict__ and v.method != "regular":
                # Don't use exact for edit distance since it's too slow
                exact = False
            rtn.append((v.id(), v.verify(tokens, index=index, exact=exact)))
        return rtn

    def verify_text(self, text, index=0, exact=False, skip_edit=False):
        """
        Verifies the watermark in the given text.

        Args:
            text: The text to verify.
            index: The index of the token to verify.
            exact: Whether to use exact matching.
            skip_edit: Whether to skip edit distance verification.

        Returns:
            A list of tuples containing the verifier ID and the verification result.
        """
        tokens = self.tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt"
        ).to(self.rng.device)
        return self.verify(
            tokens, index=index, exact=exact, skip_edit=skip_edit
        )
