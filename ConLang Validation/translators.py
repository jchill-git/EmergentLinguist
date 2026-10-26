"""Translators turn an annotated plain-language sentence into an annotated conlang sentence.

A translator receives a conllu.TokenList (UD-annotated English) and returns a
new TokenList for the conlang. Because the annotation travels with each word,
the tests can run on the conlang output without re-parsing it.

To add a translator: subclass Translator, implement translate(), and decorate
it with @register_translator("name"). It is then selectable via
`run_validation.py --translator name`.
"""

import hashlib
import random
from abc import ABC, abstractmethod

from conllu.models import Token, TokenList

TRANSLATORS = {}


def register_translator(name: str):
    def decorator(cls):
        cls.name = name
        TRANSLATORS[name] = cls
        return cls
    return decorator


def get_translator(name: str, **kwargs) -> "Translator":
    if name not in TRANSLATORS:
        raise KeyError(f"Unknown translator '{name}'. Available: {sorted(TRANSLATORS)}")
    return TRANSLATORS[name](**kwargs)


class Translator(ABC):
    name = "base"

    @abstractmethod
    def translate(self, sentence: TokenList) -> TokenList:
        """Return the translated sentence with ids and heads renumbered to the new order."""

    def translate_all(self, sentences: list) -> list:
        return [self.translate(s) for s in sentences]

    @staticmethod
    def reorder(sentence: TokenList, new_order: list) -> TokenList:
        """Rebuild a sentence in new_order (a list of old token ids), remapping ids and heads.

        The original id is kept in misc as OrigId so translations can be traced back.
        """
        old_to_new = {old_id: new_idx + 1 for new_idx, old_id in enumerate(new_order)}
        by_id = {tok["id"]: tok for tok in sentence}
        tokens = []
        for old_id in new_order:
            tok = Token(dict(by_id[old_id]))
            tok["id"] = old_to_new[old_id]
            tok["head"] = 0 if tok["head"] == 0 else old_to_new[tok["head"]]
            tok["misc"] = {**(tok["misc"] or {}), "OrigId": str(old_id)}
            tokens.append(tok)
        return TokenList(tokens, metadata=dict(sentence.metadata))


# UD relations (without subtypes) whose position the dummy grammar decides.
UD_RELATIONS = [
    "nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "vocative", "expl",
    "dislocated", "advcl", "advmod", "discourse", "aux", "cop", "mark", "nmod",
    "appos", "nummod", "acl", "amod", "det", "clf", "case", "conj", "cc", "fixed",
    "flat", "compound", "list", "parataxis", "orphan", "goeswith", "reparandum", "dep",
]


@register_translator("dummy")
class DummyTranslator(Translator):
    """A toy conlang: a random but consistent word order plus a word cipher.

    Word order: the seed fixes, for every UD relation, whether a dependent goes
    before or after its head and in what order dependents on the same side
    appear. The sentence is then rebuilt from the dependency tree, so phrases
    stay together (e.g. "the black case" moves as a unit). Punctuation stays last.

    Cipher: each English word maps to a made-up word built from syllables,
    chosen by a hash so the same word always gets the same cipher word.
    """

    ONSETS = ["p", "t", "k", "m", "n", "s", "l", "r", "v", "z", "sh", "th"]
    VOWELS = ["a", "e", "i", "o", "u"]

    def __init__(self, seed: int = 0, cipher: bool = True, directions: dict = None):
        """directions optionally overrides the sampled grammar, e.g. {"obj": "before"}."""
        self.seed = seed
        self.cipher = cipher
        rng = random.Random(seed)
        self.directions = {rel: rng.choice(["before", "after"]) for rel in UD_RELATIONS}
        self.directions.update(directions or {})
        self.priority = {rel: rng.random() for rel in UD_RELATIONS}

    def _placement(self, deprel: str):
        base = deprel.split(":")[0]
        if base == "punct":
            return "after", 2.0  # after every other dependent
        return self.directions.get(base, "after"), self.priority.get(base, 1.0)

    def linearize(self, sentence: TokenList) -> list:
        children = {}
        for tok in sentence:
            children.setdefault(tok["head"], []).append(tok)

        def visit(tok_id):
            before, after = [], []
            for child in children.get(tok_id, []):
                side, prio = self._placement(child["deprel"])
                (before if side == "before" else after).append((prio, child["id"]))
            order = []
            for _, child_id in sorted(before):
                order.extend(visit(child_id))
            order.append(tok_id)
            for _, child_id in sorted(after):
                order.extend(visit(child_id))
            return order

        order = []
        for _, root_id in sorted((self._placement("root")[1], t["id"]) for t in children.get(0, [])):
            order.extend(visit(root_id))
        return order

    def cipher_word(self, word: str) -> str:
        if not any(c.isalnum() for c in word):
            return word
        digest = hashlib.md5(f"{self.seed}:{word.lower()}".encode()).digest()
        n_syllables = 1 + digest[0] % 3
        return "".join(
            self.ONSETS[digest[1 + 2 * i] % len(self.ONSETS)]
            + self.VOWELS[digest[2 + 2 * i] % len(self.VOWELS)]
            for i in range(n_syllables)
        )

    def translate(self, sentence: TokenList) -> TokenList:
        out = self.reorder(sentence, self.linearize(sentence))
        if self.cipher:
            for tok in out:
                tok["misc"]["Orig"] = tok["form"]
                tok["form"] = self.cipher_word(tok["form"])
                tok["lemma"] = self.cipher_word(tok["lemma"] or tok["form"])
        out.metadata["text"] = " ".join(tok["form"] for tok in out)
        return out
