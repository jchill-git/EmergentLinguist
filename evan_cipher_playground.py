"""Evan's cipher playground: try the word translation, and write the LLM test prompt.

The cipher now lives in conlang_transformation_functions.py, in the section marked
WORD TRANSLATION (Evan), and sentence_transform() runs it as its last step.

To see the setup and some translations, run:
    .venv/bin/python evan_cipher_playground.py
To write the LLM test prompt (llm_test_prompt.txt) and its answer key, run:
    .venv/bin/python evan_cipher_playground.py prompt
"""

import random
import sys
from collections import Counter

from conlang_transformation_functions import (
    TRIGRAM_MAPPING, nlp, sentence_transform, translate_word,
)

EXAMPLE_LINES = range(0, 10)   # Sample_Passages.txt lines 1-10: Jon and the wolf, Samuel the sailor
TEST_LINES = range(10, 16)     # lines 11-16: the traveler with the saxophone
PROMPT_SEED = 0                # fixes the random clause order of causatives

PROMPT_INTRO = """\
We are working on a Rosetta Stone style translation problem.

Below is a text in English, followed by the same text in an invented language. \
The two versions match paragraph by paragraph, and within each paragraph sentence \
by sentence. The invented language has its own vocabulary, word order and grammar, \
so words do not line up one-to-one with English.

Study the parallel text, then translate the new passage at the end from the \
invented language into English. Keep the paragraph numbering. After your \
translation, briefly list the rules of the language you worked out."""


def translate_paragraph(paragraph: str) -> str:
    """Translate a paragraph sentence by sentence, with the dashes removed."""
    sentences = [s.text.strip() for s in nlp(paragraph).sents if s.text.strip()]
    return " ".join(sentence_transform(s) for s in sentences).replace("-", "")


def write_test_prompt(prompt_path="llm_test_prompt.txt", answer_path="llm_test_answer_key.txt"):
    """Write the LLM prompt, and the English of the test passage as an answer key."""
    random.seed(PROMPT_SEED)
    with open("Sample_Passages.txt", encoding="utf-8") as f:
        paragraphs = [line.strip() for line in f if line.strip()]
    example = [paragraphs[i] for i in EXAMPLE_LINES]
    test = [paragraphs[i] for i in TEST_LINES]

    def numbered(texts):
        return "\n\n".join(f"{n}. {t}" for n, t in enumerate(texts, 1))

    prompt = "\n\n".join([
        PROMPT_INTRO,
        "=== Text in English ===", numbered(example),
        "=== The same text in the invented language ===", numbered([translate_paragraph(p) for p in example]),
        "=== New passage to translate into English ===", numbered([translate_paragraph(p) for p in test]),
    ])
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt + "\n")
    with open(answer_path, "w", encoding="utf-8") as f:
        f.write(numbered(test) + "\n")


if __name__ == "__main__" and sys.argv[1:] == ["prompt"]:
    write_test_prompt()
    print("Wrote llm_test_prompt.txt and llm_test_answer_key.txt")
elif __name__ == "__main__":
    lengths = Counter(len(c) for c in TRIGRAM_MAPPING.values())
    print(f"{len(TRIGRAM_MAPPING)} trigrams; chunk lengths: "
          + ", ".join(f"{n} of length {k}" for k, n in sorted(lengths.items())))
    print("Most common trigrams:", ", ".join(f"{t!r}->{c}" for t, c in list(TRIGRAM_MAPPING.items())[:10]))
    for word in ["wolf", "sheep", "shepherd", "village", "town", "star", "saxophone"]:
        print(f"  {word} -> {translate_word(word)}")
    with open("sentences.txt", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    for sentence in sentences[:10]:
        print(sentence)
        print("  ->", sentence_transform(sentence))
