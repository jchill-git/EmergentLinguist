import spacy
from dataclasses import dataclass, field, replace
from lemminflect import getInflection
import random

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

"""a
FINAL PIPELINE FUNCTION
Add your functions above and then order them appropriately in the function below.
"""
def sentence_transform(sentence):
    return render_sentence(mark_applicative(mark_accusatives(mark_causative_and_reorder(passive_to_active(yes_no_to_particle(build_token_records(nlp(sentence))))))))