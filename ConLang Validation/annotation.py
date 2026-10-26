"""Load plain-language sentences and parse them into UD annotation.

Parsing uses spacy-stanza (Stanza's UD-trained models behind the spaCy API),
since spaCy's own English models do not produce UD dependency labels.
Sentences are represented as conllu.TokenList objects throughout the suite.
Each token's misc field keeps the English lemma as EnLemma, so tests can still
recognise words like "very" after a translator has changed the forms and lemmas.
"""

from pathlib import Path

import conllu
from conllu.models import Token, TokenList

# Bump when the annotation format changes, so cached parses are redone.
ANNOTATION_VERSION = "2"


def load_sentences(path) -> list:
    """One sentence per line; blank lines are skipped."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_pipeline(lang: str = "en"):
    import spacy_stanza
    return spacy_stanza.load_pipeline(
        lang, processors="tokenize,mwt,pos,lemma,depparse", logging_level="WARN"
    )


def doc_to_tokenlists(doc, sent_id_prefix: str) -> list:
    """Convert a parsed spaCy Doc into one TokenList per sentence (ids start at 1)."""
    tokenlists = []
    for s_idx, sent in enumerate(doc.sents):
        start = sent.start
        tokens = []
        for tok in sent:
            is_root = tok.head.i == tok.i or tok.dep_.lower() == "root"
            feats = tok.morph.to_dict() or None
            tokens.append(Token({
                "id": tok.i - start + 1,
                "form": tok.text,
                "lemma": tok.lemma_,
                "upos": tok.pos_,
                "xpos": tok.tag_ or None,
                "feats": feats,
                "head": 0 if is_root else tok.head.i - start + 1,
                "deprel": "root" if is_root else tok.dep_,
                "deps": None,
                "misc": {"EnLemma": tok.lemma_},
            }))
        sent_id = sent_id_prefix if len(list(doc.sents)) == 1 else f"{sent_id_prefix}-{s_idx + 1}"
        tokenlists.append(TokenList(tokens, metadata={
            "sent_id": sent_id, "text": sent.text, "annotation_version": ANNOTATION_VERSION}))
    return tokenlists


def parse_sentences(sentences: list, nlp) -> list:
    tokenlists = []
    for i, text in enumerate(sentences, start=1):
        tokenlists.extend(doc_to_tokenlists(nlp(text), sent_id_prefix=str(i)))
    return tokenlists


def write_conllu(tokenlists: list, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(tl.serialize() for tl in tokenlists)


def read_conllu(path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return conllu.parse(f.read())


def load_or_parse(sentences_path, cache_path, lang: str = "en") -> list:
    """Parse sentences_path, reusing cache_path if it is newer than the input and current."""
    sentences_path, cache_path = Path(sentences_path), Path(cache_path)
    if cache_path.exists() and cache_path.stat().st_mtime >= sentences_path.stat().st_mtime:
        cached = read_conllu(cache_path)
        if all(s.metadata.get("annotation_version") == ANNOTATION_VERSION for s in cached):
            return cached

    tokenlists = parse_sentences(load_sentences(sentences_path), load_pipeline(lang))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_conllu(tokenlists, cache_path)
    return tokenlists
