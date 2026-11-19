"""Evan's playground: try the morphology rules on the test sentences.

The rules now live in conlang_transformation_functions.py, in the section marked
MORPHOLOGY (Evan), and sentence_transform() runs them as its last step.

To see the current translation, run:
    .venv/bin/python evan_playground.py
"""

from conlang_transformation_functions import (
    apply_morphology, build_token_records, mark_accusatives, mark_applicative, nlp,
    passive_to_active, render_sentence, sentence_transform, yes_no_to_particle,
)


def morphology_only(sentence: str) -> str:
    """sentence_transform without word order or the causative shuffle, to show the
    morphology on English word order (so causatives get no -erg)."""
    records = build_token_records(nlp(sentence))
    records = yes_no_to_particle(records)
    records = passive_to_active(records)
    records = mark_accusatives(records)
    records = mark_applicative(records)
    return render_sentence(apply_morphology(records))


if __name__ == "__main__":
    with open("sentences.txt", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    for sentence in sentences[:15]:
        print(sentence)
        print("  morphology only:", morphology_only(sentence))
        print("  full pipeline:  ", sentence_transform(sentence))
