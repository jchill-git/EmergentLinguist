"""Tests for word_order.py.

Run:  python test_word_order.py        (or: pytest test_word_order.py)

Expected outputs are what each rule should produce from a correct spaCy parse.
If a test fails, first check the parse (print_parse) - it may be a spaCy
labelling error rather than a rule bug.
"""

import itertools

import spacy

from word_order import FEATURES, WordOrderState, apply_word_order

nlp = spacy.load("en_core_web_sm")


def run(text, *features):
    return apply_word_order(nlp(text), features).text()


def print_parse(text):
    for t in nlp(text):
        print(f"{t.i:>2} {t.text:<10} {t.pos_:<6} {t.tag_:<5} {t.dep_:<10} -> {t.head.i} {t.head.text}")


CASES = [
    # (features, English, expected conlang order)
    # 81A SOV
    (("81A",), "The dog ate the bone.", "The dog the bone ate."),
    (("81A",), "She gave him a book.", "She him a book gave."),
    (("81A",), "The dog is big.", "The dog big is."),
    (("81A",), "He did not eat the bone.", "He the bone eat did not."),
    (("81A",), "He picked up the ball.", "He the ball up picked."),
    (("81A",), "Did you see the bird?", "you the bird see Did?"),
    (("81A",), "What did you see?", "you What see did?"),
    (("81A",), "He said that the dog ate the bone.", "He that the dog the bone ate said."),
    # 85A Postpositions
    (("85A",), "The cat sat on the mat.", "The cat sat the mat on."),
    (("85A",), "The house is near the river.", "The house is the river near."),
    (("85A",), "The bone in the box is old.", "the box in The bone is old."),
    # 86A Genitive-Noun
    (("86A",), "The roof of the house is red.", "the house The roof is red."),
    (("86A",), "John's dog barked.", "John's dog barked."),
    (("86A",), "I drank a cup of tea.", "I drank a cup of tea."),
    # 87A Noun-Adjective
    (("87A",), "A big brown dog barked.", "A dog big brown barked."),
    (("87A",), "The very old man smiled.", "The man very old smiled."),
    (("87A",), "The dog is big.", "The dog is big."),
    # 90A Relative clause-Noun
    (("90A",), "The man who saw the dog was happy.", "saw the dog The man was happy."),
    (("90A",), "The book that I read was long.", "I read The book was long."),
    # All five together
    (tuple(FEATURES), "The old man who owns the dog sat on the bench.",
     "the dog owns The man old the bench on sat."),
    (tuple(FEATURES), "The sailor's wife waited for him at the end of the road.",
     "The sailor's wife him for the road the end at waited."),
    (("81A", "85A"), "She bought a book for her son.", "She her son for a book bought."),
    # punctuation and edge cases
    (("81A",), "Every morning, he fed the dog.", "he Every morning, the dog fed."),
    (("90A",), "The dog, which had brown fur, loved the water.",
     "had brown fur The dog loved the water."),
    (("81A", "87A"), "When the storm ended, the sailor, tired and happy, went home.",
     "When the storm ended, the sailor, tired and happy, home went."),
    (("86A",), "The sailor gave the girl half of the gold.",
     "The sailor gave the girl half of the gold."),
]

# Known spaCy parse errors (documented in the README, not rule bugs):
#   "The man who saw the dog left."  -> spaCy reads "the dog left" as a clause
#   under "saw", so "left" is not the main verb.

SENTENCES_FOR_ORDER_TEST = [c[1] for c in CASES] + [
    "The tall woman who lived in the old house of the miller gave her son a red apple.",
    "Yesterday the wolf that the hunter saw came closer to the sheep.",
]


def test_cases():
    failed = 0
    for feats, en, expected in CASES:
        got = run(en, *feats)
        ok = got == expected
        failed += not ok
        print(f"[{'ok' if ok else 'FAIL'}] {'+'.join(feats):<24} {en}\n        -> {got}"
              + ("" if ok else f"\n   expected {expected}"))
    assert failed == 0, f"{failed} case(s) failed"


def test_any_order_gives_same_result():
    """The five functions are independent: every order gives the same output."""
    for en in SENTENCES_FOR_ORDER_TEST:
        doc = nlp(en)
        results = set()
        for perm in itertools.permutations(FEATURES):
            state = WordOrderState.from_doc(doc)
            for fid in perm:
                state = FEATURES[fid](state)
            results.add(tuple(state.order()))
        assert len(results) == 1, f"order-dependent result for: {en}"
    print(f"[ok] all 120 orders agree on {len(SENTENCES_FOR_ORDER_TEST)} sentences")


def test_input_not_modified():
    doc = nlp("The old man saw the dog in the garden.")
    s0 = WordOrderState.from_doc(doc)
    before = s0.text()
    for f in FEATURES.values():
        f(s0)
    assert s0.text() == before


def test_tags_kept():
    state = apply_word_order(nlp("The dogs ate the bones."))
    ate = [t for t in state.tokens() if t.text == "ate"][0]
    assert ate.lemma_ == "eat" and "Tense=Past" in str(ate.morph)


def test_every_token_kept_once():
    for en in SENTENCES_FOR_ORDER_TEST:
        state = apply_word_order(nlp(en))
        order = state.order()
        assert sorted(order + sorted(state.dropped)) == list(range(len(state.doc)))


if __name__ == "__main__":
    test_cases()
    test_any_order_gives_same_result()
    test_input_not_modified()
    test_tags_kept()
    test_every_token_kept_once()
    print("all tests passed")
