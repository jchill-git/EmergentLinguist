import spacy
import hashlib
import re
import string
from collections import Counter
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from lemminflect import getInflection
import random
from typing import Callable, Iterable

nlp = spacy.load("en_core_web_sm")

"""
Token-wise data structure built from a spaCy Doc, for tracking edits
(markers, reordering) without touching the original parse.

Setup:
    pip install spacy
    python -m spacy download en_core_web_sm
"""
@dataclass
class TokenRecord:
    surface: str          # original surface form
    lemma: str
    original_index: int   # position in the original sentence (never changes)
    pos: str              # coarse part of speech (UPOS), e.g. "VERB"
    tag: str              # fine-grained tag, e.g. "VBD"
    dep: str              # dependency label, e.g. "nsubj"
    markers: list[str] = field(default_factory=list)  # new markers added to this token
    current_index: int = -1                           # position in the edited sentence
    head_index: int = -1  # original_index of this token's syntactic head (a root points to itself)


def build_token_records(doc) -> list[TokenRecord]:
    """
    Convert a spaCy Doc (one sentence) into a list of TokenRecords.
    `current_index` starts equal to `original_index` and is meant to be
    updated as tokens are moved; `markers` starts empty.
    """
    base = doc[0].i if len(doc) else 0  # keeps indices sentence-relative for spans
    return [
        TokenRecord(
            surface=tok.text,
            lemma=tok.lemma_,
            original_index=i,
            pos=tok.pos_,
            tag=tok.tag_,
            dep=tok.dep_,
            current_index=i,
            head_index=tok.head.i - base,
        )
        for i, tok in enumerate(doc)
    ]

"""
Render a list of TokenRecords back into a sentence string, tidying the
capitalization and punctuation that reordering or deleting tokens can garble.

Setup:
    pip install spacy
    python -m spacy download en_core_web_sm
(also needs token_records.py, and mark_accusative_records.py for the demo)
"""

TERMINAL = {".", "!", "?", "…", "..."}
WEAK = {",", ";", ":"}
CLOSERS = {")", "]", "}", "”"}
NO_SPACE_BEFORE = {".", ",", ";", ":", "!", "?", ")", "]", "}", "%", "...", "…", "”", "’"}
NO_SPACE_AFTER = {"(", "[", "{", "“", "‘"}
CLITICS = {"n't", "n’t", "'s", "’s", "'re", "’re", "'ve", "’ve", "'ll", "’ll", "'d", "’d", "'m", "’m"}


# --------------------------------------------------------------------------
# Punctuation repair
# --------------------------------------------------------------------------
def _is_punct(rec: TokenRecord) -> bool:
    return rec.pos == "PUNCT" or rec.surface in (TERMINAL | WEAK | NO_SPACE_BEFORE | NO_SPACE_AFTER)


def _move_final_punctuation(tokens):
    """Keep the punctuation that originally ended the sentence at the end."""
    originals = sorted((r for r in tokens if r.original_index >= 0), key=lambda r: r.original_index)
    tail = []
    for rec in reversed(originals):
        if not _is_punct(rec):
            break
        tail.insert(0, rec)
    if not tail:
        return tokens
    tail_ids = {id(r) for r in tail}
    return [r for r in tokens if id(r) not in tail_ids] + tail


def _clean_punctuation(tokens):
    """Drop stray commas/semicolons/colons and duplicated terminal marks."""
    kept = []
    for i, rec in enumerate(tokens):
        prev = kept[-1].surface if kept else None
        nxt = tokens[i + 1].surface if i + 1 < len(tokens) else None
        if rec.surface in WEAK:
            if prev is None or prev in WEAK or prev in TERMINAL or prev in NO_SPACE_AFTER:
                continue  # nothing for it to follow
            if nxt is None or nxt in WEAK or nxt in TERMINAL or nxt in CLOSERS:
                continue  # nothing for it to precede
        elif rec.surface in TERMINAL and prev == rec.surface:
            continue      # duplicated ". ."
        kept.append(rec)
    return kept


# --------------------------------------------------------------------------
# Capitalization repair
# --------------------------------------------------------------------------
def _is_ordinary_capital(rec: TokenRecord) -> bool:
    """A capitalized word that is only capitalized for being sentence-initial."""
    s = rec.surface
    return (
        s[:1].isupper()
        and rec.pos != "PROPN"
        and rec.tag not in ("NNP", "NNPS")
        and s != "I"
        and not (len(s) > 1 and s.isupper())  # acronyms such as NASA
    )


def _surfaces(tokens, fix_capitalization):
    """Map id(record) -> the surface form to print."""
    surfaces = {id(r): r.surface for r in tokens}
    if not fix_capitalization:
        return surfaces
    words = [r for r in tokens if any(c.isalpha() for c in r.surface)]
    if not words:
        return surfaces
    first = words[0]
    # The originally sentence-initial word loses its capital if it has moved
    for rec in tokens:
        if rec is not first and rec.original_index == 0 and _is_ordinary_capital(rec):
            surfaces[id(rec)] = rec.surface[0].lower() + rec.surface[1:]
    # Whatever word now opens the sentence gets the capital
    s = first.surface
    i = next(i for i, c in enumerate(s) if c.isalpha())
    surfaces[id(first)] = s[:i] + s[i].upper() + s[i + 1 :]
    return surfaces


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _attaches_to_previous(prev: TokenRecord, rec: TokenRecord) -> bool:
    """True if no space belongs between `prev` and `rec` (punctuation, clitics)."""
    return (
        rec.surface in NO_SPACE_BEFORE
        or rec.surface.lower() in CLITICS
        or prev.surface in NO_SPACE_AFTER
    )


def render_sentence(
    records: list[TokenRecord],
    marker_sep: str = "",
    wrap: tuple = ("[", "]"),
    fix_capitalization: bool = True,
    fix_punctuation: bool = True,
) -> str:
    """
    Build a sentence from TokenRecords. The records are not modified.

    - Tokens are placed in order of `current_index` (ties keep their list order).
    - Each marker in a token's `markers` is wrapped in `wrap` and appended to
      its surface form: "mouse[accMarker]". Several markers give
      "mouse[accMarker][defMarker]". `marker_sep` goes between the token and its
      markers (default "": attached); pass wrap=("", "") for unbracketed markers.
    - Spaces are omitted before punctuation and contraction clitics.

    fix_punctuation (default True):
    - punctuation that originally ended the sentence is kept at the end;
    - commas/semicolons/colons left with nothing to separate (at the start,
      end, or next to other punctuation) are dropped, as are duplicate
      terminal marks.

    fix_capitalization (default True):
    - the word that now opens the sentence is capitalized;
    - the originally sentence-initial word is lowercased if it moved, unless it
      is a proper noun, "I" or an acronym.
    """
    tokens = sorted(records, key=lambda r: r.current_index)
    if fix_punctuation:
        tokens = _clean_punctuation(_move_final_punctuation(tokens))
    surfaces = _surfaces(tokens, fix_capitalization)

    out, prev = "", None
    for rec in tokens:
        if prev is not None and not _attaches_to_previous(prev, rec):
            out += " "
        markers = (f"{wrap[0]}{m}{wrap[1]}" for m in rec.markers)
        out += marker_sep.join([surfaces[id(rec)], *markers])
        prev = rec
    return out

"""
Add an accusative marker to the TokenRecords of direct objects.

Setup:
    pip install spacy
    python -m spacy download en_core_web_sm
(also needs token_records.py in the same folder)
"""

ACCUSATIVE_DEPS = ("dobj", "obj")  # spaCy's English models use "dobj"; UD-style models use "obj"


def mark_accusatives(
    records: list[TokenRecord],
    marker: str = "accMarker",
    include_pronouns: bool = True,
) -> list[TokenRecord]:
    """
    Append `marker` to the `markers` list of every token that is a direct
    object (the head word of an accusative noun phrase). Modifies the records
    in place and also returns them.

    include_pronouns=False restricts marking to full (non-pronominal) noun
    phrases, so "She saw him" is left unmarked but "She saw the dog" is marked.
    """
    for rec in records:
        if rec.dep not in ACCUSATIVE_DEPS:
            continue
        if rec.pos == "PRON" and not include_pronouns:
            continue
        if marker not in rec.markers:
            rec.markers.append(marker)
    return records

"""
Rewrite passive clauses as active, working on TokenRecords.

Input:  a list of TokenRecords (from token_records.build_token_records)
Output: a new list of TokenRecords for the active sentence. Tokens keep their
        original_index; current_index gives their new position (0..n-1), so
        render_sentence() can print the result. Dependency labels and heads are
        updated for the tokens whose roles change.

Setup:
    pip install spacy lemminflect
    python -m spacy download en_core_web_sm
(also needs token_records.py)
"""

FINITE = {"VBZ", "VBP", "VBD"}
TO_OBJECT = {"i": "me", "he": "him", "she": "her", "we": "us", "they": "them"}
TO_SUBJECT = {"me": "I", "him": "he", "her": "she", "us": "we", "them": "they"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _copy(rec):
    return replace(rec, markers=list(rec.markers))


def _subtree(records, idx):
    """original_index of `idx` and of every token below it in the dependency tree."""
    seen, stack = set(), [idx]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(r.original_index for r in records if r.head_index == cur and r.original_index >= 0)
    return seen


def _word_agreement(word):
    """Person/number implied by a pronoun or placeholder word: '1sg', '3sg' or 'pl'."""
    if word in ("i", "me"):
        return "1sg"
    if word in ("we", "us", "they", "them", "you", "people"):
        return "pl"
    return "3sg"


def _finite_form(lemma, past, agr):
    """Re-agreed finite form of be / have / do."""
    if lemma == "be":
        return ("was" if agr != "pl" else "were") if past else {"1sg": "am", "3sg": "is", "pl": "are"}[agr]
    if lemma == "have":
        return "had" if past else ("has" if agr == "3sg" else "have")
    return "did" if past else ("does" if agr == "3sg" else "do")


def _finite_tag(past, agr):
    return "VBD" if past else ("VBZ" if agr == "3sg" else "VBP")


# --------------------------------------------------------------------------
# Convert one passive clause
# --------------------------------------------------------------------------
def _convert_clause(records, subj_id, placeholder_agent):
    """Return a new record list with one passive clause made active, or None."""
    ordered = sorted(records, key=lambda r: r.current_index)
    pos = {id(r): i for i, r in enumerate(ordered)}
    by_id = {r.original_index: r for r in ordered if r.original_index >= 0}

    def kids(idx):
        return [r for r in ordered if r.head_index == idx and r.original_index != idx]

    subj = by_id.get(subj_id)
    verb = by_id.get(subj.head_index) if subj else None
    if subj is None or subj.dep != "nsubjpass" or verb is None:
        return None
    verb_kids = kids(verb.original_index)
    auxpass = next((c for c in verb_kids if c.dep == "auxpass" and c.lemma == "be"), None)
    if auxpass is None:
        return None
    agent = next((c for c in verb_kids if c.dep == "agent"), None)
    pobj = next((c for c in kids(agent.original_index) if c.dep == "pobj"), None) if agent else None

    patient_ids = _subtree(ordered, subj.original_index)
    patient = [r for r in ordered if r.original_index in patient_ids]
    if pos[id(patient[0])] > pos[id(verb)]:
        return None  # questions, fronted agents, etc.: leave untouched
    at_start = pos[id(patient[0])] == 0

    # New subject: the "by" agent, or a placeholder for a truncated passive
    if pobj is not None:
        if pos[id(agent)] < pos[id(verb)]:
            return None
        np_ids = _subtree(ordered, pobj.original_index)
        new_subj = [_copy(r) for r in ordered if r.original_index in np_ids]
        plural = pobj.tag in ("NNS", "NNPS") or any(c.dep == "conj" for c in kids(pobj.original_index))
        agr = "pl" if plural else _word_agreement(pobj.surface.lower())
        if len(new_subj) == 1 and new_subj[0].surface.lower() in TO_SUBJECT:
            new_subj[0].surface = TO_SUBJECT[new_subj[0].surface.lower()]
        for r in new_subj:
            if r.original_index == pobj.original_index:
                r.dep, r.head_index = "nsubj", verb.original_index
    else:
        if placeholder_agent is None:
            return None  # truncated passive: left unchanged
        agr = _word_agreement(placeholder_agent.lower())
        new_subj = [TokenRecord(placeholder_agent, placeholder_agent.lower(), -1, "PRON", "NN",
                                "nsubj", [], -1, verb.original_index)]
    if at_start:
        new_subj[0].surface = new_subj[0].surface[0].upper() + new_subj[0].surface[1:]

    # Old subject (patient) -> object
    new_pat = [_copy(r) for r in patient]
    if len(new_pat) == 1 and new_pat[0].surface.lower() in TO_OBJECT:
        new_pat[0].surface = TO_OBJECT[new_pat[0].surface.lower()]
    elif at_start and new_pat[0].pos != "PROPN":
        new_pat[0].surface = new_pat[0].surface[0].lower() + new_pat[0].surface[1:]
    for r in new_pat:
        if r.original_index == subj.original_index:
            r.dep = "dobj"

    # Verb group
    past = auxpass.tag == "VBD"
    negated = any(c.dep == "neg" for c in verb_kids + kids(auxpass.original_index))
    lemma = verb.lemma

    def inflect(tag):
        found = getInflection(lemma, tag=tag)
        return found[0] if found else verb.surface

    auxpass_replacement = None
    if auxpass.tag in FINITE:                     # simple tense: "was read"
        if negated:                               # do-support: "did not read"
            auxpass_replacement = replace(
                auxpass, surface=_finite_form("do", past, agr), lemma="do",
                tag=_finite_tag(past, agr), dep="aux", markers=list(auxpass.markers))
            verb_surface, verb_tag = lemma, "VB"
        else:
            verb_tag = _finite_tag(past, agr)
            verb_surface = inflect(verb_tag)
    elif auxpass.tag == "VBG":                    # "is being read" -> "is reading"
        verb_surface, verb_tag = inflect("VBG"), "VBG"
    elif auxpass.tag == "VB":                     # "will be read" -> "will read"
        verb_surface, verb_tag = lemma, "VB"
    else:                                         # "has been read" -> "has read"
        verb_surface, verb_tag = verb.surface, "VBN"
    new_verb = replace(verb, surface=verb_surface, tag=verb_tag, markers=list(verb.markers))

    # Re-agree finite be/have/do auxiliaries with the new subject
    aux_swaps = {}
    for aux in verb_kids:
        if aux.dep == "aux" and aux.tag in FINITE and aux.lemma in ("be", "have", "do"):
            aux_past = aux.tag == "VBD"
            surface = _finite_form(aux.lemma, aux_past, agr)
            if aux.surface[0].isupper():
                surface = surface.capitalize()
            aux_swaps[id(aux)] = replace(aux, surface=surface, tag=_finite_tag(aux_past, agr),
                                         markers=list(aux.markers))

    # Assemble the new sentence
    patient_set = {id(r) for r in patient}
    agent_set = {id(r) for r in ordered if r.original_index in _subtree(ordered, agent.original_index)} if agent else set()
    out, subject_placed = [], False
    for r in ordered:
        if id(r) in patient_set:
            if not subject_placed:
                out.extend(new_subj)
                subject_placed = True
        elif id(r) in agent_set:
            continue
        elif r is auxpass:
            if auxpass_replacement is not None:
                out.append(auxpass_replacement)
        elif r is verb:
            out.append(new_verb)
            out.extend(new_pat)
        else:
            out.append(aux_swaps.get(id(r)) or _copy(r))

    for r in out:                                 # re-attach children of the removed auxiliary
        if auxpass_replacement is None and r.head_index == auxpass.original_index:
            r.head_index = verb.original_index
    for i, r in enumerate(out):
        r.current_index = i
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def passive_to_active(records: list[TokenRecord], placeholder_agent: str | None = None) -> list[TokenRecord]:
    """
    Return a new list of TokenRecords with every passive clause made active.
    The input list is not modified.

    Truncated passives (no "by"-agent) are left unchanged unless
    `placeholder_agent` is given (e.g. "someone"), which becomes the subject.
    """
    current = [_copy(r) for r in records]
    for subj_id in [r.original_index for r in records if r.dep == "nsubjpass"]:
        converted = _convert_clause(current, subj_id, placeholder_agent)
        if converted is not None:
            current = converted
    return current

def mark_applicative(records: list[TokenRecord]) -> list[TokenRecord]:
    """
    Add an Applicative Marker to the verb in English
    applicative-type sentences (double-object constructions).

    A construction is identified by a verb with two objects:
        - one dative object (indirect object)
        - one direct object

    The Applicative Marker is added to the verb.

    Returns the modified list of TokenRecords.
    """

    dative_deps = {"dative", "iobj"}
    direct_object_deps = {"dobj", "obj"}

    # Group objects according to their syntactic head
    objects_by_head = {}

    for record in records:
        if record.dep in dative_deps | direct_object_deps:
            objects_by_head.setdefault(record.head_index, []).append(record)

    # Examine each verb's objects
    for head_index, objects in objects_by_head.items():

        # Find the verb governing these objects
        verb = next(
            (r for r in records if r.original_index == head_index),
            None
        )

        if verb is None or verb.pos not in {"VERB", "AUX"}:
            continue

        # Separate dative and direct objects
        dative_objects = [
            obj for obj in objects
            if obj.dep in dative_deps
        ]

        direct_objects = [
            obj for obj in objects
            if obj.dep in direct_object_deps
        ]

        # Require exactly one object of each type
        if len(dative_objects) == 1 and len(direct_objects) == 1:

            # Add the Applicative Marker to the verb
            if "Applicative Marker" not in verb.markers:
                verb.markers.append("appMarker")

    return records

CAUSATIVE_VERBS = {"make", "have", "get", "cause", "force", "let", "allow"}
CLAUSE_DEPS = {"xcomp", "ccomp"}


def _is_question_particle(rec: TokenRecord) -> bool:
    """The particle added by yes_no_to_particle(): a new token (original_index -1) labelled 'discourse'."""
    return rec.original_index < 0 and rec.dep == "discourse"


def mark_causative_and_reorder(records: list[TokenRecord]) -> list[TokenRecord]:
    """
    Identify periphrastic causative constructions, add a
    Causative Marker to the causative verb, and randomly
    reorder the main and embedded clauses.

    Causative verbs include:
        make, have, get, cause, force, let, allow

    A construction is identified by a causative verb with
    an xcomp or ccomp dependent.

    A question particle at the front of the sentence is not part of either
    clause, so it is set aside before shuffling and put back at the front.

    Returns the reordered list of TokenRecords.
    """
    ordered = sorted(records, key=lambda r: r.current_index)

    # Set the question particle aside so it can't be shuffled with the main clause
    particle = [r for r in ordered if _is_question_particle(r)]
    body = [r for r in ordered if not _is_question_particle(r)]

    # Map each token's original_index to the records that depend on it
    children = {}
    for rec in body:
        children.setdefault(rec.head_index, []).append(rec)

    def subtree(head):
        """ids of `head` and every record below it (records are tracked by identity,
        because new tokens all share original_index -1)."""
        seen, stack = set(), [head]
        while stack:
            rec = stack.pop()
            if id(rec) in seen:
                continue
            seen.add(id(rec))
            if rec.original_index >= 0:
                stack.extend(children.get(rec.original_index, []))
        return seen

    for verb in body:
        if verb.lemma.lower() not in CAUSATIVE_VERBS:
            continue

        # Embedded clause heads belonging to this verb
        embedded_heads = [
            r for r in body
            if r.head_index == verb.original_index
            and r is not verb
            and r.dep in CLAUSE_DEPS
        ]
        if not embedded_heads:
            continue

        if "Causative Marker" not in verb.markers:
            verb.markers.append("causMarker")

        # One group of records per embedded clause; everything else is the main clause
        embedded_clauses, embedded_ids = [], set()
        for head in embedded_heads:
            ids = subtree(head)
            embedded_ids |= ids
            embedded_clauses.append([r for r in body if id(r) in ids])
        main_clause = [r for r in body if id(r) not in embedded_ids]

        clauses = [main_clause, *embedded_clauses]
        random.shuffle(clauses)

        # The question particle always comes first
        reordered = particle + [r for clause in clauses for r in clause]
        for new_index, rec in enumerate(reordered):
            rec.current_index = new_index
        return reordered

    # No causative construction found
    return records

"""
Replace the word order / do-support that marks a yes-no question with a
question particle at the start of the sentence.

    Did she bake a cake?   ->  [question token] She baked a cake?
    Is the cat sleeping?   ->  [question token] The cat is sleeping?
    Can you swim?          ->  [question token] You can swim?
"""

FINITE = {"VBZ", "VBP", "VBD"}
WH_TAGS = {"WDT", "WP", "WP$", "WRB"}
NEG_CLITICS = {"n't", "n’t"}


def yes_no_to_particle(
    records: list[TokenRecord],
    question_token: str = "[questionToken]",
    require_question_mark: bool = True,
) -> list[TokenRecord]:
    """
    Return a new list of TokenRecords in which a yes-no question has been put
    in declarative word order and given `question_token` as its first token.
    The input is not modified.

    - Do-support is removed and the main verb re-inflected ("Did she bake" ->
      "she baked"), unless the sentence is negated, where "do" stays
      ("Didn't she bake" -> "she didn't bake").
    - Any other inverted auxiliary, modal or copula is moved back after the
      subject ("Is the cat happy" -> "the cat is happy").
    - Wh-questions, declaratives and anything else are returned unchanged.
    - By default a "?" must be present to treat the sentence as a question.
    - The final punctuation is left as it was.
    """
    unchanged = [_copy(r) for r in records]
    ordered = sorted(records, key=lambda r: r.current_index)
    words = [r for r in ordered if r.pos != "PUNCT"]
    if not words:
        return unchanged
    if require_question_mark and not any(r.surface == "?" for r in ordered):
        return unchanged
    if any(r.tag in WH_TAGS for r in ordered):
        return unchanged  # wh-questions are not yes-no questions

    # The sentence must open with a finite auxiliary / copula / modal
    first = words[0]
    if first.dep not in ("ROOT", "aux", "auxpass"):
        return unchanged
    if not (first.tag == "MD" or (first.lemma in ("be", "have", "do") and first.tag in FINITE)):
        return unchanged

    pos = {id(r): i for i, r in enumerate(ordered)}
    by_id = {r.original_index: r for r in ordered if r.original_index >= 0}

    def kids(idx):
        return [r for r in ordered if r.head_index == idx and r.original_index != idx]

    verb = first if first.dep == "ROOT" else by_id.get(first.head_index)
    if verb is None:
        return unchanged

    # Subject-auxiliary inversion: the subject must come after the auxiliary
    subj = next((c for c in kids(verb.original_index) if c.dep in ("nsubj", "nsubjpass")), None)
    if subj is None:
        return unchanged
    subject_ids = _subtree(ordered, subj.original_index)
    subject = [r for r in ordered if r.original_index in subject_ids]
    if pos[id(subject[0])] < pos[id(first)]:
        return unchanged

    root = next((r for r in ordered if r.dep == "ROOT"), verb)
    particle = TokenRecord(question_token, question_token, -1, "PART", "RP",
                           "discourse", [], 0, root.original_index)

    negated = any(c.dep == "neg" for c in kids(verb.original_index) + kids(first.original_index))
    do_support = first.lemma == "do" and first.dep == "aux" and not negated

    out = [particle]
    if do_support:
        # "Did she bake" -> "she baked": drop "do", move its tense onto the verb
        found = getInflection(verb.lemma, tag=first.tag)
        new_verb = replace(verb, surface=found[0] if found else verb.surface,
                           tag=first.tag, markers=list(verb.markers))
        for r in ordered:
            if r is first:
                continue
            out.append(new_verb if r is verb else _copy(r))
        for r in out:  # re-attach anything that hung off the removed "do"
            if r.head_index == first.original_index:
                r.head_index = verb.original_index
    else:
        # Un-invert: put the auxiliary (and an attached n't) after the subject
        moved = [first] + [r for r in ordered
                           if r.dep == "neg" and r.surface in NEG_CLITICS and pos[id(r)] == pos[id(first)] + 1]
        moved_ids = {id(r) for r in moved}
        for r in ordered:
            if id(r) in moved_ids:
                continue
            out.append(_copy(r))
            if r is subject[-1]:
                out.extend(_copy(m) for m in moved)

    for i, r in enumerate(out):
        r.current_index = i
    return out


# ==========================================================================
# WORD ORDER (Janet): 81A, 85A, 86A, 87A, 90A
# ==========================================================================
"""
Word order transformations for the CPSC 532 conlang (Group 2: Word Order, Janet).

One function per highlighted WALS feature, all working on Joe's TokenRecord
list (defined at the top of this file):

    81A  sov_81A                  Order of Subject, Object and Verb      -> SOV
    85A  postpositions_85A        Order of Adposition and Noun Phrase    -> Postpositions
    86A  genitive_noun_86A        Order of Genitive and Noun             -> Genitive-Noun
    87A  noun_adjective_87A       Order of Adjective and Noun            -> Noun-Adjective
    90A  relcl_noun_90A           Order of Relative Clause and Noun      -> Relative clause-Noun
    apply_word_order              all five, for sentence_transform()

Input and output (every function)
---------------------------------
    f(records: list[TokenRecord], **options) -> list[TokenRecord]

Input:  TokenRecords in their current order (by ``current_index``), e.g.
        straight from build_token_records(nlp(sentence)) or from an earlier
        pipeline step. Fields read: surface, lemma, pos, tag, dep,
        original_index, head_index, current_index.
Output: a NEW list of copied TokenRecords (the input is not modified), sorted
        and renumbered so ``current_index`` is 0..n-1 in conlang order.
        surface, lemma, pos, tag, dep, head_index, original_index and markers
        are unchanged, so later steps (markers, morphology, render_sentence)
        can still use them. Words the conlang does not write ("of" in a
        genitive, "who/which/that" in a relative clause, the commas around a
        non-restrictive relative clause) are left out of the returned list,
        the same way passive_to_active() removes "by".

How it works
------------
The dependency tree is rebuilt from ``head_index``. Each function only
reorders the direct dependents of one kind of head (verbs for 81A,
adpositions for 85A, nouns for 86A/87A/90A), and every dependent moves together
with its whole phrase. Nothing is re-parsed, so tags stay correct after words
move. Inside a noun phrase the order is fixed by rank:

    [genitive] [noun-modifying PP] [relative clause] [the/a, compounds] NOUN [adjectives] [other]

so the five functions give the same result in any order. Records added by other steps (original_index -1, e.g.
the question particle from yes_no_to_particle or a placeholder subject from
passive_to_active) are handled like any other word attached to their head; the
question particle keeps its place at the start of the sentence.

"""


NOUN_POS = {"NOUN", "PROPN", "PRON"}
VERB_POS = {"VERB", "AUX"}


# --------------------------------------------------------------------------
# Tree view of a TokenRecord list (internal)
# --------------------------------------------------------------------------

class _Tree:
    """The records as a dependency tree. ``layout[id(r)]`` lists ``r`` and its
    direct dependents in their current order; functions edit layouts, then
    ``emit()`` writes the new order back to TokenRecords."""

    def __init__(self, records):
        self.ordered = sorted(records, key=lambda r: r.current_index)
        self.pos = {id(r): n for n, r in enumerate(self.ordered)}
        by_orig = {r.original_index: r for r in self.ordered if r.original_index >= 0}
        self.parent = {}
        for r in self.ordered:
            p = by_orig.get(r.head_index)
            if p is not None and p is not r:
                self.parent[id(r)] = p
        self.children = {id(r): [] for r in self.ordered}
        for r in self.ordered:
            if id(r) in self.parent:
                self.children[id(self.parent[id(r)])].append(r)
        self.dropped = set()
        self.reattach_dep = {}  # id(dropped record) -> new dep for its children
        # Order siblings by where their phrase starts now (robust even if an
        # earlier step left a phrase split up).
        self.start = {}
        for r in self.ordered:
            self.start[id(r)] = min(self.pos[id(x)] for x in self.subtree(r))
        self.layout = {}
        for r in self.ordered:
            items = [r] + self.children[id(r)]
            self.layout[id(r)] = sorted(
                items, key=lambda x: self.pos[id(x)] if x is r else self.start[id(x)])

    def subtree(self, r):
        out, stack = [], [r]
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(self.children[id(cur)])
        return out

    def kids(self, r):
        """Direct dependents of ``r`` in current English order."""
        return sorted(self.children[id(r)], key=lambda x: self.start[id(x)])

    def head(self, r):
        return self.parent.get(id(r))

    def place(self, head, child, side, rank_fn):
        """Move ``child`` (with its phrase) to ``side`` ('before'/'after') of
        ``head``; that side is then ordered by ``rank_fn`` (stable)."""
        items = [x for x in self.layout[id(head)] if x is not child]
        h = next(n for n, x in enumerate(items) if x is head)
        before, after = items[:h], items[h + 1:]
        if side == "before":
            before = sorted(before + [child], key=rank_fn)
        else:
            after = sorted(after + [child], key=rank_fn)
        self.layout[id(head)] = before + [head] + after

    def emit(self):
        out = []

        def expand(r):
            for x in self.layout[id(r)]:
                if x is r:
                    out.append(r)
                else:
                    expand(x)

        roots = sorted((r for r in self.ordered if id(r) not in self.parent),
                       key=lambda r: self.start[id(r)])
        for root in roots:
            expand(root)
        result = []
        for r in out:
            if id(r) in self.dropped:
                continue
            new = replace(r, markers=list(r.markers))
            new.current_index = len(result)
            # A word whose head was left out ("the year" under a dropped "of")
            # is re-attached to the nearest written ancestor, so later steps
            # still see one tree.
            p = self.parent.get(id(r))
            if p is not None and id(p) in self.dropped:
                new.dep = self.reattach_dep.get(id(p), new.dep)
                while p is not None and id(p) in self.dropped:
                    p = self.parent.get(id(p))
                if p is not None:
                    new.head_index = p.original_index
            result.append(new)
        return result


# Rank of a dependent placed before a noun (lower = earlier).
def _rank_before(t: _Tree):
    def rank(r):
        if r.dep == "poss" or _is_of_genitive(t, r):
            return 10                       # genitive first
        if _is_adposition(r) and t.head(r) is not None and t.head(r).pos in {"NOUN", "PROPN"}:
            return 15                       # noun-modifying PP
        if r.dep == "relcl":
            return 20                       # relative clause
        return 50                           # the/a, compounds, ... keep English order
    return rank


def _rank_after(r):
    return 10 if r.dep == "amod" else 50    # adjectives right after the noun


# --------------------------------------------------------------------------
# 81A  Order of Subject, Object and Verb  ->  SOV
# --------------------------------------------------------------------------

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
OBJECT_DEPS = {"dobj", "obj", "attr", "acomp", "oprd", "ccomp", "xcomp"}
DATIVE_DEPS = {"dative", "iobj"}
VERB_COMPLEX_DEPS = {"aux", "auxpass", "neg"}
PARTICLE_DEPS = {"prt"}
# Clause-edge words keep their English side (before or after the clause).
EDGE_DEPS = {"cc", "conj", "mark", "advcl", "intj", "parataxis", "preconj",
             "meta", "appos", "dep", "discourse"}


def sov_81A(records, *, obliques_before_object: bool = True,
            auxiliaries_after_verb: bool = True):
    """81A: put every clause in Subject-Object-Verb order.

    Input
        records : list[TokenRecord]
            Reads ``pos`` (VERB/AUX heads a clause), ``dep`` (each dependent's
            role) and ``tag`` (WRB), plus head_index / current_index.
        obliques_before_object : bool, default True
            Where prepositional phrases, adverbs and other non-core dependents
            go. True gives S-X-O-V (WALS 84A "XOV"); False gives S-O-X-V.
        auxiliaries_after_verb : bool, default True
            True puts auxiliaries and "not" after the main verb ("will eat" ->
            "eat will"), the usual order in OV languages. Not one of our WALS
            rows: the group should confirm.

    Output
        New list[TokenRecord]. Inside every clause headed by a VERB/AUX:
            [clause-initial edge words] subject [obliques] [indirect object]
            [object / complement clause] [particle] VERB [aux, neg]
            [clause-final edge words]
        Each slot keeps its English internal order. Wh-words stay in the slot
        of their role ("what did you see?" -> "you what see did?").

    Labels used
        subjects   nsubj, nsubjpass, csubj, csubjpass, expl
        objects    dobj / obj, attr, acomp (copula complements), oprd, ccomp, xcomp
        indirect   dative / iobj that is a noun phrase ("gave him a book")
        verb group aux, auxpass, neg; prt kept right before the verb
        edges      cc, conj, mark, advcl, discourse (question particle) ... keep their side
        punct      clause-final . ? ! stay last; a comma moves with the phrase it followed
        anything else (prep, agent, advmod, npadvmod, ...) = oblique

    Example
        "The dog ate the bone in the garden." -> "The dog in the garden the bone ate."
    """
    t = _Tree(records)
    for v in t.ordered:
        if v.pos not in VERB_POS:
            continue
        kids = t.kids(v)
        if not kids:
            continue
        vpos = t.pos[id(v)]
        front, subj, obl, dat, obj, prt, vgroup, end = ([] for _ in range(8))
        glued = {}
        for n, c in enumerate(kids):
            d, cpos = c.dep, t.start[id(c)]
            if d == "punct":
                later_words = [k for k in kids[n + 1:] if k.dep != "punct"]
                before = [k for k in kids[:n] if k.dep != "punct"]
                prev_key = max([(t.start[id(k)], k) for k in before] + ([(vpos, v)] if vpos < cpos else []),
                               key=lambda p: p[0], default=None)
                if cpos > vpos and all(k.dep in EDGE_DEPS for k in later_words):
                    end.append(c)                   # clause-final . ? !, or the comma
                                                    # before ", and ..." / ", but ..."
                elif prev_key is None:
                    front.append(c)                 # opening mark, e.g. a quote
                elif prev_key[1] is v:
                    end.append(c)                   # mark right after the verb
                else:
                    glued.setdefault(id(prev_key[1]), []).append(c)
            elif d in EDGE_DEPS or (d == "advmod" and c.tag == "WRB" and v.dep == "advcl"):
                (front if cpos < vpos else end).append(c)
            elif d in SUBJECT_DEPS:
                subj.append(c)
            elif d in OBJECT_DEPS:
                obj.append(c)
            elif d in DATIVE_DEPS and c.pos != "ADP":
                dat.append(c)                       # bare indirect object
            elif d in PARTICLE_DEPS:
                prt.append(c)
            elif d in VERB_COMPLEX_DEPS:
                vgroup.append(c)
            else:
                obl.append(c)                       # incl. "for him", "to her"
        core = (obl + dat + obj) if obliques_before_object else (dat + obj + obl)
        if auxiliaries_after_verb:
            order = front + subj + core + prt + [v] + vgroup + end
        else:
            order = front + subj + core + vgroup + prt + [v] + end
        t.layout[id(v)] = [x for r in order for x in [r] + glued.get(id(r), [])]
    return t.emit()


# --------------------------------------------------------------------------
# 85A  Order of Adposition and Noun Phrase  ->  Postpositions
# --------------------------------------------------------------------------

ADPOSITION_OBJECT_DEPS = {"pobj", "pcomp"}

# "a cup of tea", "half of the gold": "of" here is not a possessor.
QUANTITY_NOUNS = {
    "cup", "glass", "bottle", "bowl", "piece", "slice", "bit", "lot", "kind",
    "sort", "type", "number", "group", "part", "couple", "pair", "bunch",
    "rest", "amount", "plenty", "handful", "majority", "most", "some", "all",
    "half", "dozen", "pile", "series", "variety", "pinch", "drop",
}


def _is_adposition(r) -> bool:
    """A preposition heading a phrase: dep prep / agent, or dative with pos ADP
    (spaCy labels "for him", "to her" as dative)."""
    return r.dep in {"prep", "agent"} or (r.dep == "dative" and r.pos == "ADP")


def _is_of_genitive(t: _Tree, r) -> bool:
    """The "of" of a possessive "X of Y" (both nouns). Shared by 85A and 86A."""
    if r.surface.lower() != "of" or r.dep != "prep":
        return False
    head = t.head(r)
    if head is None or head.pos not in {"NOUN", "PROPN"} or head.lemma.lower() in QUANTITY_NOUNS:
        return False
    return any(c.dep == "pobj" and c.pos in NOUN_POS for c in t.children[id(r)])


def postpositions_85A(records, *, adnominal_pp_before_noun: bool = True):
    """85A: turn prepositions into postpositions.

    Input
        records : list[TokenRecord]
            Reads ``dep`` (prep / agent / dative and their pobj / pcomp),
            ``pos`` of the preposition and of the word its phrase attaches to.
        adnominal_pp_before_noun : bool, default True
            A prepositional phrase that modifies a noun ("the bone in the box")
            also moves before that noun, like other noun modifiers in a
            head-final language: "the box in the bone". "of"-genitives are left
            to 86A.

    Output
        New list[TokenRecord] where every adposition follows its object:
            "in the house" -> "the house in". Stranded prepositions (no object)
            and phrasal-verb particles (prt, "pick up") are not changed.

    Example
        "She put the cup on the table." -> "She put the cup the table on."
    """
    t = _Tree(records)
    rank = _rank_before(t)
    for p in t.ordered:
        if not _is_adposition(p):
            continue
        objs = [c for c in t.kids(p) if c.dep in ADPOSITION_OBJECT_DEPS]
        if not objs:
            continue
        others = [x for x in t.layout[id(p)] if x is not p and all(x is not o for o in objs)]
        before = [x for x in others if t.start[id(x)] < t.pos[id(p)]]
        after = [x for x in others if t.start[id(x)] > t.pos[id(p)]]
        t.layout[id(p)] = before + objs + [p] + after
        head = t.head(p)
        if (adnominal_pp_before_noun and head is not None and head.pos in {"NOUN", "PROPN"}
                and not _is_of_genitive(t, p)):
            t.place(head, p, "before", rank)
    return t.emit()


# --------------------------------------------------------------------------
# 86A  Order of Genitive and Noun  ->  Genitive-Noun
# --------------------------------------------------------------------------

def genitive_noun_86A(records, *, drop_of: bool = True, drop_possessive_marker: bool = False):
    """86A: put every possessor before the possessed noun.

    Input
        records : list[TokenRecord]
            Reads ``dep`` (poss, prep + pobj, case), ``surface``, ``lemma``, ``pos``.
        drop_of : bool, default True
            Leave "of" out of "X of Y" genitives. Set False if Group 1 wants to
            keep it (e.g. to turn it into a genitive ending).
        drop_possessive_marker : bool, default False
            Also leave out the "'s" of "John's" (dep case).

    Output
        New list[TokenRecord]:
            "the roof of the house" -> "the house the roof"
            "John's dog"            -> unchanged (already Genitive-Noun)
        The possessor goes first in its noun phrase. Partitive "of" ("a cup of
        tea", "half of the gold") is not a genitive.
        When "of" is dropped, its object is re-attached to the possessed noun
        with dep "poss" (head_index = that noun), the same label as "John's",
        so a later step can mark every genitive the same way.

    Example
        "The roof of the house is red." -> "the house The roof is red."
    """
    t = _Tree(records)
    rank = _rank_before(t)
    for r in t.ordered:
        if _is_of_genitive(t, r):
            t.place(t.head(r), r, "before", rank)
            if drop_of:
                t.dropped.add(id(r))
                t.reattach_dep[id(r)] = "poss"  # "the year" becomes the possessor of "hour"
        elif r.dep == "poss" and t.head(r) is not None:
            t.place(t.head(r), r, "before", rank)
        elif r.dep == "case" and drop_possessive_marker:
            t.dropped.add(id(r))
    return t.emit()


# --------------------------------------------------------------------------
# 87A  Order of Adjective and Noun  ->  Noun-Adjective
# --------------------------------------------------------------------------

def noun_adjective_87A(records):
    """87A: put attributive adjectives after the noun.

    Input
        records : list[TokenRecord]
            Reads ``dep`` == "amod" and the head's ``pos``.

    Output
        New list[TokenRecord] where every adjective phrase directly follows its
        noun, keeping the English order of several adjectives and of words
        attached to them ("very big", "big and brown"):
            "a big brown dog" -> "a dog big brown"
        Predicate adjectives ("the dog is big") belong to the clause (81A).
        Adjectives already after the noun stay where they are.

    Example
        "The old man saw a small bird." -> "The man old saw a bird small."
    """
    t = _Tree(records)
    for r in t.ordered:
        head = t.head(r)
        if (r.dep == "amod" and head is not None and head.pos in NOUN_POS
                and t.start[id(r)] < t.pos[id(head)]):
            t.place(head, r, "after", _rank_after)
    return t.emit()


# --------------------------------------------------------------------------
# 90A  Order of Relative Clause and Noun  ->  Relative clause-Noun
# --------------------------------------------------------------------------

RELATIVIZER_TAGS = {"WDT", "WP"}  # which, that, who, whom (not "whose" = WP$)


def relcl_noun_90A(records, *, drop_relativizer: bool = True):
    """90A: put relative clauses before the noun they modify.

    Input
        records : list[TokenRecord]
            Reads ``dep`` == "relcl" and the ``tag`` of the words directly under
            the relative clause verb.
        drop_relativizer : bool, default True
            Leave out who / whom / which / that when it is a direct dependent of
            the relative clause verb (prenominal relative clauses usually have
            no relative pronoun).

    Output
        New list[TokenRecord] where every relative clause comes before its noun
        (after a genitive, before articles):
            "the man who saw the dog" -> "saw the dog the man"
        With 81A also applied the clause itself is SOV: "the dog saw the man".
        The commas around a non-restrictive clause are left out too.
        Kept as is: "whose", where / when, "in which".

    Example
        "I know the man who owns the dog." -> "I know owns the dog the man."
    """
    t = _Tree(records)
    rank = _rank_before(t)
    for r in t.ordered:
        head = t.head(r)
        if r.dep != "relcl" or head is None:
            continue
        t.place(head, r, "before", rank)
        if drop_relativizer:
            for c in t.children[id(r)]:
                if c.tag in RELATIVIZER_TAGS:
                    t.dropped.add(id(c))
        # commas just before / at the end of / just after the clause
        span = sorted(t.pos[id(x)] for x in t.subtree(r))
        for p in (span[0] - 1, span[-1], span[-1] + 1):
            if 0 <= p < len(t.ordered):
                x = t.ordered[p]
                if x.surface == "," and t.head(x) in (r, head):
                    t.dropped.add(id(x))
    return t.emit()


# --------------------------------------------------------------------------
# Pipeline wrapper
# --------------------------------------------------------------------------

WORD_ORDER_FEATURES: dict[str, Callable] = {
    "81A": sov_81A,
    "85A": postpositions_85A,
    "86A": genitive_noun_86A,
    "87A": noun_adjective_87A,
    "90A": relcl_noun_90A,
}


def apply_word_order(records, features: Iterable[str] = tuple(WORD_ORDER_FEATURES)):
    """Run the chosen word order functions (default: all five).

    Input:  list[TokenRecord]
    Output: new list[TokenRecord] in conlang word order (current_index 0..n-1).
    """
    for fid in features:
        records = WORD_ORDER_FEATURES[fid](records)
    return records

# ==========================================================================
# MORPHOLOGY (Evan): 30A, 31A, 33A, 34A, 35A, 37A, 38A, 44A, 46A, 49A, 51A,
#                    52A, 53A, 57A, 98A; spells the 109A and 111A markers
# ==========================================================================
"""
Morphology and nominal categories for the CPSC 532 conlang (Evan).

One section per WALS feature (features that describe the same rule share one),
all run by apply_morphology on Joe's TokenRecord list:

    30A, 31A        _gender_30A_31A            Two genders, not based on sex
    33A, 34A        _plural_prefix_33A_34A     Plural prefix, on all nouns
    35A, 44A        _pronouns_35A_44A          One pronoun "it", plural "xea-it"
    37A, 38A        _articles_37A_38A          "the" -> ending, no "a/an"
    46A             _indefinites_46A           "someone" -> "who-do"
    49A, 51A,
    52A, 98A        _case_suffixes_49A_51A_52A_98A   Six case endings, accusative
    53A             _ordinals_53A              "second" -> "taka-two"
    57A             _possessive_prefixes_57A   "her son" -> "a-son"
    109A, 111A      _spell_markers             appMarker -> "bhu-", causMarker -> "-erg"
    apply_morphology                           all of the above, for sentence_transform()

41A (two-way demonstratives) and 27A (no reduplication) need no code: "this" and
"that" stay separate words, and nothing is repeated.

Input and output
----------------
    apply_morphology(records: list[TokenRecord]) -> list[TokenRecord]

Input:  TokenRecords in their current order (by ``current_index``), after every
        other step of sentence_transform(). Fields read: surface, lemma, pos,
        tag, dep, original_index, head_index, current_index, markers.
Output: a NEW list of copied TokenRecords (the input is not modified), in the
        same order, renumbered so ``current_index`` is 0..n-1.
        ``surface`` is the conlang spelling; pos, tag and dep are unchanged.
        English words that became endings or prefixes ("the", "a/an", "in",
        "of", "'s", "his" ...) are left out of the list, and any word that
        depended on one is re-attached to its head, so every head_index still
        points to a word in the list.
        The markers from other steps are spelled out and removed: [questionToken],
        appMarker and causMarker become words and affixes, and accMarker is
        dropped, since the -qo ending replaces it.

How it works
------------
It runs last, after word order. Word order needs the English words ("the",
"of", "in") to find phrases, and 86A has already turned "X of Y" into a
genitive with dep "poss", which the case step marks with -co. Every rule reads
the English spelling of a word from a copy taken at the start, so the order of
the spelling changes does not change which rules apply. The order does decide
where affixes go: whole-word changes first, then endings ("sky-mi-zh"), then
prefixes ("a-xea-son", "oa-taka-two").
"""


# --------------------------------------------------------------------------
# Helpers (internal)
# --------------------------------------------------------------------------

class _Words:
    """The records of one sentence, with the English spelling of each word kept
    aside. The rules below change spellings one after another, so every rule
    reads the English word from here, never from ``surface``."""

    def __init__(self, records):
        self.records = sorted(records, key=lambda r: r.current_index)
        self._english = {id(r): r.surface.lower() for r in self.records}
        self._by_index = {r.original_index: r for r in self.records if r.original_index >= 0}

    def english(self, rec):
        """The word's English spelling in lowercase, even after it has been changed."""
        return self._english[id(rec)]

    def head(self, rec):
        """The word this word depends on, or None for the root."""
        head = self._by_index.get(rec.head_index)
        return None if head is rec else head

    def children(self, rec):
        """The words that depend on this word. Words added by other steps
        (original_index -1, e.g. the question particle) have none."""
        if rec.original_index < 0:
            return []
        return [r for r in self.records if r.head_index == rec.original_index and r is not rec]


def _replace_word(rec, form):
    """Swap in a different word, keeping the original's capital if it had one."""
    rec.surface = form[0].upper() + form[1:] if rec.surface[:1].isupper() else form


def _add_prefix(rec, prefix):
    """Add a prefix. render_sentence capitalizes whichever word ends up first, so a
    capital that only marked the start of the English sentence is dropped here."""
    form = rec.surface
    if rec.original_index == 0 and _is_ordinary_capital(rec):
        form = form[0].lower() + form[1:]
    rec.surface = prefix + form


def _add_suffix(rec, suffix):
    rec.surface += suffix


def _is_noun(rec):
    return rec.pos in {"NOUN", "PROPN"}


def _remove_words(records, removed):
    """Leave the removed words out. Any word that depended on one moves to its head."""
    removed_ids = {id(r) for r in removed}
    head_of = {r.original_index: r.head_index for r in removed}
    kept = []
    for rec in records:
        if id(rec) in removed_ids:
            continue
        while rec.head_index in head_of and head_of[rec.head_index] != rec.head_index:
            rec.head_index = head_of[rec.head_index]
        rec.current_index = len(kept)
        kept.append(rec)
    return kept


# --------------------------------------------------------------------------
# 30A, 31A  Number of Genders  ->  Two; Sex-based Gender  ->  Non-sex-based
# --------------------------------------------------------------------------
# Each noun's gender is a coin flip made once, so a noun always keeps the same
# gender. It shows on "the" (DEFINITE_SUFFIXES) and on adjectives
# (ADJECTIVE_PREFIXES). An adjective copies the gender of the word it describes.
# "it" and "xea-it" share one coin flip, and an adjective that describes no word
# uses a flip on the adjective.

GENDERS = ("1", "2")
ADJECTIVE_PREFIXES = {"1": "ia-", "2": "oa-"}


def _gender_of(rec):
    digest = hashlib.md5(rec.lemma.lower().encode()).digest()
    return GENDERS[digest[0] % len(GENDERS)]


def _subject_of(verb, words):
    """The subject of a verb. For "who" in "the man who is old", that's the man."""
    subjects = [c for c in words.children(verb) if c.dep in {"nsubj", "nsubjpass"}]
    if subjects and subjects[0].tag not in RELATIVIZER_TAGS:
        return subjects[0]
    # The relative word ("who") points back to the noun, and 90A may have dropped it.
    if verb.dep == "relcl":
        return words.head(verb)
    return None


def _described_word(adjective, words):
    """The word an adjective describes, or None if it doesn't describe any word."""
    head = words.head(adjective)
    if head is None:
        return None
    if adjective.dep == "conj" and head.pos == "ADJ":           # "old and strong"
        return _described_word(head, words)
    if adjective.dep in {"amod", "acl"}:                        # "the old man"
        return head
    if adjective.dep in {"acomp", "attr"}:                      # "the man is old", "looks happy"
        return _subject_of(head, words)
    if adjective.dep in {"oprd", "xcomp", "ccomp", "advmod"}:   # "holds her hand high"
        objects = [c for c in words.children(head) if c.dep in {"dobj", "obj"}]
        return objects[0] if objects else _subject_of(head, words)
    return None


def _gender_30A_31A(words):
    """Give each adjective the prefix for the gender of the word it describes."""
    for rec in words.records:
        if rec.pos == "ADJ":
            gender = _gender_of(_described_word(rec, words) or rec)
            _add_prefix(rec, ADJECTIVE_PREFIXES[gender])


# --------------------------------------------------------------------------
# 33A, 34A  Coding of Nominal Plurality  ->  Plural prefix;
#           Occurrence of Nominal Plurality  ->  All nouns, always obligatory
# --------------------------------------------------------------------------
# A plural noun goes back to its English base word and takes the prefix, so
# "stars" becomes "xea-star". This includes nouns after numbers: "three xea-tree".

PLURAL_PREFIX = "xea-"


def _plural_prefix_33A_34A(words):
    """Put the plural prefix on every plural noun."""
    for rec in words.records:
        if _is_noun(rec) and rec.tag in {"NNS", "NNPS"}:
            _replace_word(rec, PLURAL_PREFIX + rec.lemma)


# --------------------------------------------------------------------------
# 35A, 44A  Plurality in Pronouns  ->  Person stem + nominal plural affix;
#           Gender in Pronouns  ->  No gender distinctions
# --------------------------------------------------------------------------
# he, she and it are all "it", and they is "it" with the noun plural: "xea-it".
# Object forms are the same word with a case ending, so "him" becomes "it-qo" and
# "to them" becomes "xea-it-ph". Other pronouns (I, you, this, who) stay in English.

PRONOUN = "it"
SINGULAR_PRONOUNS = {"he", "him", "she", "her", "it"}
PLURAL_PRONOUNS = {"they", "them"}


def _is_changed_pronoun(rec, words):
    """True for the English pronouns listed above, but not for "her" in "her son"."""
    return (rec.pos == "PRON" and rec.tag != "PRP$"
            and words.english(rec) in SINGULAR_PRONOUNS | PLURAL_PRONOUNS)


def _pronouns_35A_44A(words):
    """Turn he, him, she, her and it into "it", and they and them into "xea-it"."""
    for rec in words.records:
        if _is_changed_pronoun(rec, words):
            plural = words.english(rec) in PLURAL_PRONOUNS
            _replace_word(rec, PLURAL_PREFIX + PRONOUN if plural else PRONOUN)
            rec.lemma = PRONOUN  # so they all share one gender coin flip


# --------------------------------------------------------------------------
# 37A, 38A  Definite Articles  ->  Definite affix;
#           Indefinite Articles  ->  No indefinite, but definite article
# --------------------------------------------------------------------------
# "the" becomes an ending on its noun and "a/an" is dropped. Nouns with no
# article in English, such as names, "her son" or "food", get no ending. The
# ending matches the noun's gender (30A).

DEFINITE_SUFFIXES = {"1": "-mi", "2": "-mo"}
ARTICLES = {"the": "definite", "a": "indefinite", "an": "indefinite"}


def _articles_37A_38A(words):
    """Turn "the" into an ending on its noun and drop "a/an". Returns the removed articles."""
    # An article with no head word is the whole sentence, so it stays.
    articles = [r for r in words.records
                if r.dep == "det" and r.lemma.lower() in ARTICLES and words.head(r)]
    # A noun gets one ending, even with two articles: "the first and the last man".
    nouns = {id(words.head(a)): words.head(a)
             for a in articles if ARTICLES[a.lemma.lower()] == "definite"}
    for noun in nouns.values():
        _add_suffix(noun, DEFINITE_SUFFIXES[_gender_of(noun)])
    return articles


# --------------------------------------------------------------------------
# 46A  Indefinite Pronouns  ->  Interrogative-based
# --------------------------------------------------------------------------
# "something" is "what" with an ending, and "someone" is "who" with the same
# ending. "no one" keeps its "no": "no who-do". Like pronouns, these take case
# endings. "nothing" and "nobody" stay in English, because they are one word in
# English but would be two here.

INDEFINITE_SUFFIX = "-do"
INDEFINITES = {
    "something": "what", "anything": "what",
    "someone": "who", "somebody": "who", "anyone": "who", "anybody": "who",
}


def _indefinite_word(rec, words):
    """The new spelling of an indefinite word, or None for any other word."""
    form = words.english(rec)
    base = INDEFINITES.get(form)
    if form == "one" and any(words.english(c) == "no" for c in words.children(rec)):
        base = "who"
    return base + INDEFINITE_SUFFIX if base else None


def _indefinites_46A(words):
    """Turn "something" into "what-do", and "someone" and "no one" into "who-do"."""
    for rec in words.records:
        indefinite = _indefinite_word(rec, words)
        if indefinite:
            _replace_word(rec, indefinite)
            rec.lemma = indefinite  # so "someone" and "no one" share a coin flip


# --------------------------------------------------------------------------
# 49A, 51A, 52A, 98A  Number of Cases  ->  6-7 cases (counting the plain form);
#                     Position of Case Affixes  ->  Case suffixes;
#                     Comitatives and Instrumentals  ->  Identity;
#                     Alignment of Case Marking  ->  Nominative - accusative
# --------------------------------------------------------------------------
# Subjects stay plain and objects are marked. The ending goes last on the noun or
# pronoun, after -mi, and replaces the English preposition. "with" is one ending
# whether it means "with a friend" or "with a knife". All other prepositions,
# such as "near" or "for", stay (as postpositions, 85A).
#   - An owner is any noun with dep "poss": "the man's dog", and "X of Y" after
#     86A. -co is only for owners, so "of" stays when it means an amount ("a cup
#     of tea"), follows an adjective ("full of stars"), or means "about".
#   - The receiver under a verb that mark_applicative marked ("gave the child a
#     book") is a plain object (-qo), since the applicative shows its role.
#   - The person made to do something (the causee) is an object (-qo) if the
#     caused verb has no object of its own ("made him run"), and takes "to" (-ph)
#     if it does ("made him eat the soup"). No WALS feature covers this; it is the
#     common pattern.

CASE_ENDINGS = {
    "Acc": "-qo",  # object
    "Loc": "-zh",  # place: in, at, on
    "Dat": "-ph",  # goal: to, into
    "Abl": "-ko",  # source: from
    "Gen": "-co",  # owner: of, 's
    "Com": "-go",  # company or tool: with
}
PREPOSITION_CASES = {
    "in": "Loc", "at": "Loc", "on": "Loc",
    "to": "Dat", "into": "Dat",
    "from": "Abl",
    "of": "Gen",
    "with": "Com",
}
POSSESSIVE_MARKERS = {"'s", "’s", "'", "’"}


def _takes_case(rec, words):
    return _is_noun(rec) or _is_changed_pronoun(rec, words) or bool(_indefinite_word(rec, words))


def _preposition_case(preposition, words):
    """The case a preposition turns into, or None if it stays in English."""
    form = words.english(preposition)
    if form == "of":
        # The same test 86A uses: "of" between two nouns, the first not an amount.
        owned = words.head(preposition)
        if owned is None or not _is_noun(owned) or owned.lemma.lower() in QUANTITY_NOUNS:
            return None
    return PREPOSITION_CASES.get(form)


def _applicative_receivers(words):
    """ids of the receivers under a verb that mark_applicative has marked."""
    return {id(c) for verb in words.records if APPLICATIVE_MARKER in verb.markers
            for c in words.children(verb) if c.dep in {"dative", "iobj"} and c.pos != "ADP"}


def _causee_cases(words):
    """The case of each causee, by id(record).

    A causative is found the way mark_causative_and_reorder() finds one: a verb in
    CAUSATIVE_VERBS with a clause (CLAUSE_DEPS) under it. spaCy makes the causee
    either the subject of that clause ("made him eat") or the object of the
    causative verb ("forced him to eat").
    """
    cases = {}
    for verb in words.records:
        if verb.lemma.lower() not in CAUSATIVE_VERBS:
            continue
        for clause in words.children(verb):
            if clause.dep not in CLAUSE_DEPS:
                continue
            clause_kids = words.children(clause)
            causees = ([c for c in clause_kids if c.dep in {"nsubj", "nsubjpass"}]
                       or [c for c in words.children(verb) if c.dep in {"dobj", "obj"}])
            has_object = any(c.dep in {"dobj", "obj"} for c in clause_kids)
            if causees and _takes_case(causees[0], words):
                cases[id(causees[0])] = "Dat" if has_object else "Acc"
    return cases


def _assign_cases(words):
    """The case of each marked word (by id(record)), and the words those cases
    replace: the prepositions and the "'s"."""
    cases, replaced = {}, []
    receivers = _applicative_receivers(words)
    causees = _causee_cases(words)
    for rec in words.records:
        head = words.head(rec)
        if not _takes_case(rec, words):
            continue
        if id(rec) in causees:
            cases[id(rec)] = causees[id(rec)]
        elif (rec.dep == "pobj" and head is not None and head.pos == "ADP"
                and _preposition_case(head, words)):
            cases[id(rec)] = _preposition_case(head, words)
            replaced.append(head)
        elif rec.dep == "poss" and _is_noun(rec):
            cases[id(rec)] = "Gen"
            replaced.extend(c for c in words.children(rec)
                            if c.dep == "case" and words.english(c) in POSSESSIVE_MARKERS)
        elif rec.dep in {"dobj", "obj"} or id(rec) in receivers:   # "gives the man food"
            cases[id(rec)] = "Acc"
        elif rec.dep in {"dative", "iobj"}:                         # no object: "tell him"
            cases[id(rec)] = "Dat"

    # Words joined by "and" share a case, unless the later word starts its own
    # clause, as in "at home and the child is a sailor".
    for rec in words.records:
        head = words.head(rec)
        if (rec.dep == "conj" and head is not None and id(head) in cases
                and _takes_case(rec, words) and id(rec) not in cases
                and not any(c.dep in {"nsubj", "nsubjpass"} for c in words.children(rec))):
            cases[id(rec)] = cases[id(head)]
    return cases, replaced


def _case_suffixes_49A_51A_52A_98A(words):
    """Add the case endings. Returns the prepositions and "'s" they replace."""
    cases, replaced = _assign_cases(words)
    for rec in words.records:
        if id(rec) in cases:
            _add_suffix(rec, CASE_ENDINGS[cases[id(rec)]])
    return replaced


# --------------------------------------------------------------------------
# 53A  Ordinal Numerals  ->  First, two-th, three-th
# --------------------------------------------------------------------------
# "first" stays its own word, and later ones are the number with a prefix, so
# "second" becomes "taka-two". They are still adjectives: "oa-taka-two".

ORDINAL_PREFIX = "taka-"
ORDINALS = {
    "second": "two", "third": "three", "fourth": "four", "fifth": "five", "sixth": "six",
    "seventh": "seven", "eighth": "eight", "ninth": "nine", "tenth": "ten",
    "eleventh": "eleven", "twelfth": "twelve", "thirteenth": "thirteen",
    "fourteenth": "fourteen", "fifteenth": "fifteen", "sixteenth": "sixteen",
    "seventeenth": "seventeen", "eighteenth": "eighteen", "nineteenth": "nineteen",
    "twentieth": "twenty",
}


def _ordinals_53A(words):
    """Turn "second", "third" and so on into "taka-" plus the number."""
    for rec in words.records:
        if rec.pos == "ADJ" and words.english(rec) in ORDINALS:
            _replace_word(rec, ORDINAL_PREFIX + ORDINALS[words.english(rec)])


# --------------------------------------------------------------------------
# 57A  Position of Pronominal Possessive Affixes  ->  Possessive prefixes
# --------------------------------------------------------------------------
# The prefix replaces the English word and goes first on the noun, before xea-,
# so "her sons" becomes "a-xea-son". Words with no noun after them ("mine") stay.

POSSESSIVE_PREFIXES = {
    "his": "a-", "her": "a-", "its": "a-",
    "their": "xa-",
    "my": "u-",
    "our": "xu-",
    "your": "i-",  # "xi-" means several people, but English "your" doesn't show which
}


def _possessive_prefixes_57A(words):
    """Turn his, her, my and the rest into prefixes on their noun. Returns the removed words."""
    possessives = [r for r in words.records
                   if r.dep == "poss" and r.tag == "PRP$" and words.english(r) in POSSESSIVE_PREFIXES
                   and words.head(r) is not None and _is_noun(words.head(r))]
    for possessive in possessives:
        _add_prefix(words.head(possessive), POSSESSIVE_PREFIXES[words.english(possessive)])
    return possessives


# --------------------------------------------------------------------------
# Markers from other steps  ->  conlang words and affixes
# --------------------------------------------------------------------------
# Each step marks the words it finds, and this only spells the marks:
#   yes_no_to_particle          [questionToken] -> "soos", first in the sentence
#   mark_applicative            appMarker       -> "bhu-" prefix on the verb (109A)
#   mark_causative_and_reorder  causMarker      -> "-erg" suffix on the verb (111A)
#   mark_accusatives            accMarker       -> dropped, the -qo ending replaces it
# The purpose marker "-arg" (110A) is not spelled yet: no step adds a purpose marker.

QUESTION_PARTICLE = "soos"
APPLICATIVE_MARKER, APPLICATIVE_PREFIX = "appMarker", "bhu-"
CAUSATIVE_MARKER, CAUSATIVE_SUFFIX = "causMarker", "-erg"
ACCUSATIVE_MARKER = "accMarker"


def _spell_markers(words):
    """Spell the question particle, and turn the applicative and causative markers
    into their prefix and suffix."""
    for rec in words.records:
        if _is_question_particle(rec):
            rec.surface = rec.lemma = QUESTION_PARTICLE
        if APPLICATIVE_MARKER in rec.markers:
            _add_prefix(rec, APPLICATIVE_PREFIX)
        if CAUSATIVE_MARKER in rec.markers:
            _add_suffix(rec, CAUSATIVE_SUFFIX)
        rec.markers = [m for m in rec.markers
                       if m not in {APPLICATIVE_MARKER, CAUSATIVE_MARKER, ACCUSATIVE_MARKER}]


# --------------------------------------------------------------------------
# Pipeline wrapper
# --------------------------------------------------------------------------

def apply_morphology(records):
    """Run every morphology rule, in the order that puts affixes in place.

    Input:  list[TokenRecord], after every other step of sentence_transform().
    Output: new list[TokenRecord] with conlang spellings (current_index 0..n-1).

    Example
        "The sun is high in the sky." -> "Sun-mo is oa-high sky-mi-zh."
        (with word order: "Sun-mo sky-mi-zh oa-high is.")
    """
    words = _Words([_copy(r) for r in records])

    # Whole words change first, so the endings and prefixes below go on the new word.
    _plural_prefix_33A_34A(words)
    _ordinals_53A(words)
    _pronouns_35A_44A(words)
    _indefinites_46A(words)

    # These replace English words with endings or prefixes, and return the words
    # they removed. "the" comes before case, so the case ending goes last
    # ("sky-mi-zh"). The possessive comes after, so it goes first ("a-xea-son").
    removed = []
    removed += _articles_37A_38A(words)
    removed += _case_suffixes_49A_51A_52A_98A(words)
    removed += _possessive_prefixes_57A(words)

    # Last, so the adjective prefix goes first ("oa-taka-two"). The markers are
    # spelled after the cases, which read appMarker.
    _gender_30A_31A(words)
    _spell_markers(words)

    return _remove_words(words.records, removed)


# ==========================================================================
# WORD TRANSLATION (Evan): English base words -> conlang words
# ==========================================================================
"""
Word translation for the CPSC 532 conlang (Evan): a trigram cipher on the base
word of every record, run as the last step of sentence_transform().

Setup (one seed, WORD_SEED, done once when this file is imported)
    1. Collect every word in WORD_SETUP_FILES (lowercase, letters only).
       Words of 7 letters or more lose their last two letters, as in translation.
    2. Split each word into trigrams. A word whose length isn't a multiple of 3
       is padded with spaces, so "wolf" is "wol" + "f  ".
    3. The most common trigrams (found in the most words) get one letter each.
    4. The rest get a 2-letter or 4-letter chunk, drawn at random, about half of
       each as far as the 676 two-letter chunks go. Every trigram gets a
       different chunk, and no chunk spells an affix or the conlang's own words.

Translation (a fresh seed per word, from WORD_SEED and the word itself)
    shorten -> split into trigrams -> swap each for its chunk -> shuffle -> join
So a word always gets the same translation, whatever was translated before it.
    "wolf" -> "tyhleqmh", "village" -> "villa" -> "umlzrj"

Input and output
----------------
    translate_words(records: list[TokenRecord]) -> list[TokenRecord]

Input:  TokenRecords after apply_morphology(). Fields read: surface, pos.
Output: a NEW list of copied TokenRecords; only ``surface`` changes. The base word
        inside each surface is translated and its affixes stay:
        "xea-star-mo-qo" -> "xea-cdr-mo-qo". Left as they are: names (PROPN),
        punctuation, words with non-letters ("n't", numbers), and the conlang's
        own words ("it", "xea-it", "soos", "who-do", "what-do").

Known effects
-------------
Different words can share a translation: the cut merges words that differ only
in their last two letters ("return" / "returned"), and chunks of different
lengths can join into the same letters ("come" and "had" are both "yt").
"""


WORD_SEED = 0
WORD_SETUP_FILES = ("Sample_Passages.txt", "sentences.txt", "lemma_list.txt")
TRIGRAM_PAD = " "
LONG_WORD, LONG_WORD_CUT = 7, 2  # words this long or longer lose their last 2 letters

# The morphology's affixes, without their hyphens.
_AFFIX_PREFIXES = {p.strip("-") for p in [PLURAL_PREFIX, ORDINAL_PREFIX, APPLICATIVE_PREFIX,
                                          *POSSESSIVE_PREFIXES.values(), *ADJECTIVE_PREFIXES.values()]}
_AFFIX_SUFFIXES = {s.strip("-") for s in [INDEFINITE_SUFFIX, CAUSATIVE_SUFFIX,
                                          *DEFINITE_SUFFIXES.values(), *CASE_ENDINGS.values()]}
# No chunk may spell one of these, so a short translated word can't pass for an
# affix or one of the conlang's own words.
_RESERVED_CHUNKS = _AFFIX_PREFIXES | _AFFIX_SUFFIXES | {PRONOUN, QUESTION_PARTICLE}


# --------------------------------------------------------------------------
# Setup (internal)
# --------------------------------------------------------------------------

def _shorten(word):
    """Words of LONG_WORD letters or more lose their last LONG_WORD_CUT letters."""
    return word[:-LONG_WORD_CUT] if len(word) >= LONG_WORD else word


def _trigrams(word):
    """Split a word into trigrams, padding the last one with spaces."""
    word = word + TRIGRAM_PAD * (-len(word) % 3)
    return [word[i:i + 3] for i in range(0, len(word), 3)]


def _setup_words():
    """Every word in the setup files, lowercase and letters only, shortened."""
    here = Path(__file__).resolve().parent
    words = set()
    for name in WORD_SETUP_FILES:
        words.update(_shorten(w) for w in re.findall(r"[a-z]+", (here / name).read_text(encoding="utf-8").lower()))
    return sorted(words)


def _all_chunks(length):
    """Every chunk of the given length that doesn't spell a reserved word."""
    chunks = ("".join(letters) for letters in product(string.ascii_lowercase, repeat=length))
    return [c for c in chunks if c not in _RESERVED_CHUNKS]


def _build_trigram_mapping(seed=WORD_SEED):
    """Map every trigram in the setup words to its chunk, one-to-one."""
    rng = random.Random(seed)
    popularity = Counter(t for word in _setup_words() for t in set(_trigrams(word)))
    # Most common first; ties go alphabetically, so the order is the same every run.
    ranked = sorted(popularity, key=lambda t: (-popularity[t], t))

    # One letter each for the most common trigrams, as many as there are letters.
    letters = _all_chunks(1)
    rng.shuffle(letters)
    common, rest = ranked[:len(letters)], ranked[len(letters):]

    # About half 2-letter and half 4-letter, as far as the 2-letter chunks go.
    twos = _all_chunks(2)
    n_two = min(len(rest) // 2, len(twos))
    chunks = rng.sample(twos, n_two) + rng.sample(_all_chunks(4), len(rest) - n_two)
    rng.shuffle(chunks)

    mapping = dict(zip(common, letters))
    mapping.update(zip(rest, chunks))
    return mapping


TRIGRAM_MAPPING = _build_trigram_mapping()
_USED_CHUNKS = set(TRIGRAM_MAPPING.values())


def _chunk_for(trigram):
    """The trigram's chunk. A trigram the setup never saw gets a 4-letter chunk worked
    out from the trigram itself, so it is the same every time. It never reuses a setup
    chunk, but two unseen trigrams could share one."""
    if trigram in TRIGRAM_MAPPING:
        return TRIGRAM_MAPPING[trigram]
    rng = random.Random(f"{WORD_SEED}:unseen:{trigram}")
    while True:
        chunk = "".join(rng.choice(string.ascii_lowercase) for _ in range(4))
        if chunk not in _USED_CHUNKS and chunk not in _RESERVED_CHUNKS:
            return chunk


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------

def translate_word(word):
    """Translate one English word: shorten -> trigrams -> chunks -> shuffle -> join.

    Example
        "wolf" -> "tyhleqmh"
    """
    word = _shorten(word.lower())
    chunks = [_chunk_for(t) for t in _trigrams(word)]
    random.Random(f"{WORD_SEED}:word:{word}").shuffle(chunks)  # fresh seed, from the word
    return "".join(chunks)


def _translate_surface(rec):
    """Translate the base word inside a record's surface, keeping its affixes."""
    if rec.pos in {"PROPN", "PUNCT"}:
        return rec.surface
    pieces = rec.surface.split("-")
    start, end = 0, len(pieces)
    while start < end - 1 and pieces[start].lower() in _AFFIX_PREFIXES:
        start += 1
    while end - 1 > start and pieces[end - 1].lower() in _AFFIX_SUFFIXES:
        end -= 1
    base = [p.lower() for p in pieces[start:end]]
    # The conlang's own words: "it", "xea-it", "soos", "who-do", "what-do".
    if base in ([PRONOUN], [QUESTION_PARTICLE]) or (
            INDEFINITE_SUFFIX.strip("-") in pieces[end:] and base in (["who"], ["what"])):
        return rec.surface
    if not all(p.isalpha() for p in base):
        return rec.surface  # numbers, "n't" and the like stay as they are
    return "-".join(pieces[:start] + [translate_word(p) for p in base] + pieces[end:])


def translate_words(records):
    """Translate the base word of every record.

    Input:  list[TokenRecord], after apply_morphology().
    Output: new list[TokenRecord] with conlang words; only surface changes.

    Example
        "Sun-mo sky-mi-zh oa-high is." -> "Lz-mo qfpc-mi-zh oa-fok ru."
    """
    out = [_copy(r) for r in records]
    for rec in out:
        rec.surface = _translate_surface(rec)
    return out


"""a
FINAL PIPELINE FUNCTION
Add your functions above and then order them appropriately in the function below.
"""
def sentence_transform(sentence):
    records = build_token_records(nlp(sentence))
    # 1. sentence rewrites (change the English sentence itself)
    records = yes_no_to_particle(records)
    records = passive_to_active(records)
    # 2. word order: 81A, 85A, 86A, 87A, 90A (reorders phrases on the parse; no re-parsing)
    records = apply_word_order(records)
    # 3. clause shuffle for causatives (keeps each clause's internal order)
    records = mark_causative_and_reorder(records)
    # 4. markers
    records = mark_accusatives(records)
    records = mark_applicative(records)
    # 5. morphology: spells words, affixes and the markers above (reads English words)
    records = apply_morphology(records)
    # 6. word translation: English base words -> conlang words (affixes stay)
    records = translate_words(records)
    # 7. the conlang is written without dashes: "sky-mi-zh" -> "skymizh"
    return render_sentence(records).replace("-", "")


def translate_passage(text, seed=None):
    """Translate English text of any length: a sentence, a paragraph, or several.

    Paragraphs (separated by line breaks) are kept apart; within each one, spaCy
    finds the sentences and each goes through sentence_transform on its own.
    Causative sentences shuffle their clauses at random, so pass a seed (e.g. 0)
    to get the same translation every time.
    """
    if seed is not None:
        random.seed(seed)
    paragraphs = []
    for paragraph in text.splitlines():
        sentences = [s.text.strip() for s in nlp(paragraph).sents if s.text.strip()]
        paragraphs.append(" ".join(sentence_transform(s) for s in sentences))
    return "\n".join(paragraphs)
