"""
Word order transformations for the CPSC 532 conlang (Group 2: Word Order).

One function per highlighted WALS feature:

    81A  sov_81A                  Order of Subject, Object and Verb      -> SOV
    85A  postpositions_85A        Order of Adposition and Noun Phrase    -> Postpositions
    86A  genitive_noun_86A        Order of Genitive and Noun             -> Genitive-Noun
    87A  noun_adjective_87A       Order of Adjective and Noun            -> Noun-Adjective
    90A  relcl_noun_90A           Order of Relative Clause and Noun      -> Relative clause-Noun

Where this step sits in the shared pipeline
-------------------------------------------
    English text
      -> nlp(text)                    spaCy Doc (tags + dependency parse)
      -> WordOrderState.from_doc(doc)
      -> word order functions         (this file; any order, see below)
      -> state.tokens()               list[spacy.tokens.Token] in conlang order
      -> lemmatize / cipher / inflect (later steps, other groups)

Shared data structure
---------------------
Every function takes a ``WordOrderState`` and returns a NEW ``WordOrderState``
(the input is never modified). The state keeps the original spaCy ``Doc``
untouched and records, for every head token, the order of that head and its
direct dependents. A phrase is always moved together with everything under it,
so a function can never split a phrase.

Because each function only reorders the dependents of its own kind of head
(verbs for 81A, adpositions for 85A, nouns for 86A/87A/90A) and noun dependents
are placed by a fixed rank (genitive < adnominal PP < relative clause < other
pre-noun words; adjectives right after the noun), the five functions give the
same result in any order. ``test_word_order.py`` checks this.

Output for the next steps
-------------------------
``state.tokens()`` returns the real spaCy ``Token`` objects, so ``.lemma_``,
``.pos_``, ``.tag_``, ``.morph`` (tense, number, ...) and ``.dep_`` are all still
available to the morphology / cipher steps. Words removed from the surface
(e.g. "of" in a genitive, "who" in a relative clause) are listed in
``state.dropped``; they keep their tags in ``state.doc`` in case another group
wants to turn them into an affix instead.

Requires spaCy 3.x with an English pipeline (tested with en_core_web_sm 3.8.0,
which uses the ClearNLP dependency labels: nsubj, dobj, prep, pobj, amod, relcl ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from spacy.tokens import Doc, Token

__all__ = [
    "WordOrderState",
    "sov_81A",
    "postpositions_85A",
    "genitive_noun_86A",
    "noun_adjective_87A",
    "relcl_noun_90A",
    "FEATURES",
    "apply_word_order",
]


# --------------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------------

# Placement ranks for dependents moved next to a noun. Lower = closer to the
# start of its side. Words a function did not move keep DEFAULT_RANK, so they
# stay in their English order next to the noun (e.g. "the", compounds).
RANK_GENITIVE = 10        # 86A: possessor / "of"-genitive, first in the noun phrase
RANK_ADNOMINAL_PP = 15    # 85A: prepositional phrase modifying a noun
RANK_RELCL = 20           # 90A: relative clause
RANK_ADJECTIVE = 10       # 87A: first word after the noun
DEFAULT_RANK = 50

# English possessives ("John's", "his") are genitives already before the noun.
_DEFAULT_RANK_BY_DEP = {"poss": RANK_GENITIVE}

NOUN_POS = {"NOUN", "PROPN", "PRON"}
VERB_POS = {"VERB", "AUX"}


@dataclass
class WordOrderState:
    """The sentence being transformed.

    Attributes
    ----------
    doc : spacy.tokens.Doc
        The original English parse. Never modified.
    layout : dict[int, list[int]]
        For every token index ``h``: the indices of ``h`` itself and of its
        direct dependents, in conlang order. ``h`` appears in its own list.
    rank : dict[int, int]
        Placement rank given to a dependent when a function moved it next to a
        noun (see RANK_* constants).
    dropped : set[int]
        Token indices that are not written in the conlang ("of", "who", ...).
    applied : list[str]
        WALS feature IDs already applied, in order (for logging).
    """

    doc: Doc
    layout: dict[int, list[int]]
    rank: dict[int, int] = field(default_factory=dict)
    dropped: set[int] = field(default_factory=set)
    applied: list[str] = field(default_factory=list)

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_doc(cls, doc: Doc) -> "WordOrderState":
        """Input: a parsed spaCy Doc. Output: a state in plain English order."""
        layout = {}
        for tok in doc:
            layout[tok.i] = sorted([tok.i] + [c.i for c in tok.children])
        return cls(doc=doc, layout=layout)

    def copy(self) -> "WordOrderState":
        return WordOrderState(
            doc=self.doc,
            layout={h: list(v) for h, v in self.layout.items()},
            rank=dict(self.rank),
            dropped=set(self.dropped),
            applied=list(self.applied),
        )

    # ---- output -----------------------------------------------------------
    def order(self) -> list[int]:
        """Token indices in conlang order (dropped tokens excluded)."""
        out: list[int] = []

        def expand(h: int) -> None:
            for i in self.layout[h]:
                if i == h:
                    out.append(i)
                else:
                    expand(i)

        for sent in self.doc.sents:
            expand(sent.root.i)
        return [i for i in out if i not in self.dropped]

    def tokens(self) -> list[Token]:
        """spaCy Tokens in conlang order, with all their original tags."""
        return [self.doc[i] for i in self.order()]

    def text(self) -> str:
        """Readable string of the reordered sentence (for checking only)."""
        out = ""
        for tok in self.tokens():
            attach = tok.is_punct and tok.text not in "([{\"'" or tok.dep_ == "case"
            out += ("" if attach or not out else " ") + tok.text
        return out

    def records(self) -> list[dict]:
        """Plain-dict view of tokens() for logging / JSON (text, lemma, tags)."""
        return [
            {
                "i": t.i,
                "text": t.text,
                "lemma": t.lemma_,
                "pos": t.pos_,
                "tag": t.tag_,
                "dep": t.dep_,
                "head": t.head.i,
                "morph": str(t.morph),
            }
            for t in self.tokens()
        ]

    # ---- helpers used by the feature functions -----------------------------
    def _rank_of(self, i: int) -> int:
        if i in self.rank:
            return self.rank[i]
        return _DEFAULT_RANK_BY_DEP.get(self.doc[i].dep_, DEFAULT_RANK)

    def _place(self, head: int, child: int, side: str, rank: int) -> None:
        """Move ``child`` (with its whole phrase) to ``side`` ('before' or
        'after') of ``head``, at position ``rank`` on that side."""
        items = [i for i in self.layout[head] if i != child]
        h = items.index(head)
        before, after = items[:h], items[h + 1:]
        self.rank[child] = rank
        if side == "before":
            before.append(child)
            before.sort(key=self._rank_of)  # stable: unmoved words keep order
        else:
            after.append(child)
            after.sort(key=self._rank_of)
        self.layout[head] = before + [head] + after


def _children(tok: Token, deps: Iterable[str]) -> list[Token]:
    deps = set(deps)
    return [c for c in tok.children if c.dep_ in deps]


# --------------------------------------------------------------------------
# 81A  Order of Subject, Object and Verb  ->  SOV
# --------------------------------------------------------------------------

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
OBJECT_DEPS = {"dobj", "attr", "acomp", "oprd", "ccomp", "xcomp"}
DATIVE_DEPS = {"dative"}
VERB_COMPLEX_DEPS = {"aux", "auxpass", "neg"}
PARTICLE_DEPS = {"prt"}
# Clause-edge words keep their English side (before or after the clause).
EDGE_DEPS = {"punct", "cc", "conj", "mark", "advcl", "intj", "parataxis",
             "preconj", "meta", "appos", "dep"}


def sov_81A(state: WordOrderState, *, obliques_before_object: bool = True,
            auxiliaries_after_verb: bool = True) -> WordOrderState:
    """81A: put every clause in Subject-Object-Verb order.

    Input
        state : WordOrderState
            Needs, per token: ``pos_`` (to find verbs/auxiliaries heading a
            clause) and ``dep_`` (to find each dependent's role).
        obliques_before_object : bool, default True
            Where prepositional phrases, adverbs and other non-core dependents
            of the verb go. True gives S-X-O-V (WALS 84A "XOV"); False gives
            S-O-X-V ("OXV").
        auxiliaries_after_verb : bool, default True
            True puts auxiliaries and "not" after the main verb ("will eat" ->
            "eat will"), the usual order in OV languages. Not a WALS row in our
            table: the group should confirm.

    Output
        New WordOrderState. Inside every clause headed by a VERB/AUX:
            [clause-initial edge words] subject [obliques] [indirect object]
            [object / complement clause] [particle] VERB [aux, neg]
            [clause-final edge words]
        Each slot keeps its English internal order. Subject-auxiliary
        inversion in questions is undone automatically, and wh-words stay in
        the slot of the role they fill ("what did you see?" -> "you what see did?").

    spaCy labels used
        subjects   nsubj, nsubjpass, csubj, csubjpass, expl
        objects    dobj, attr, acomp (copula complements), oprd, ccomp, xcomp
        indirect   dative
        verb group aux, auxpass, neg; prt (phrasal-verb particle, kept right before the verb)
        edges      cc, conj, mark, advcl, ... keep their English side
        punct      clause-final . ? ! stay last; a comma moves with the phrase it
                   followed ("Once upon a time, ..." keeps its comma)
        anything else (prep, agent, advmod, npadvmod, ...) = oblique

    Example
        "The dog ate the bone in the garden."
        -> "The dog in the garden the bone ate."
    """
    new = state.copy()
    for tok in state.doc:
        if tok.pos_ not in VERB_POS:
            continue
        kids = list(tok.children)
        if not kids:
            continue
        front, subj, obl, dat, obj, prt, vgroup, end = ([] for _ in range(8))
        glued: dict[int, list[int]] = {}  # comma -> stays right after the phrase it followed
        for n, c in enumerate(kids):  # children come in English order
            d = c.dep_
            if d == "punct":
                later_words = [k for k in kids[n + 1:] if k.dep_ != "punct"]
                # the word just before this mark: the verb or one of its dependents
                before = [k.i for k in kids[:n] if k.dep_ != "punct"]
                if tok.i < c.i:
                    before.append(tok.i)
                prev = max(before) if before else None
                if c.i > tok.i and not later_words:
                    end.append(c.i)            # clause-final . ? ! stays last
                elif prev is None:
                    front.append(c.i)          # opening mark, e.g. a quote
                elif prev == tok.i:
                    end.append(c.i)            # mark right after the verb
                else:
                    glued.setdefault(prev, []).append(c.i)
            elif d in EDGE_DEPS or (d == "advmod" and c.tag_ == "WRB"
                                    and tok.dep_ == "advcl"):
                # "When the storm ended, ...": "when" introduces the clause
                (front if c.i < tok.i else end).append(c.i)
            elif d in SUBJECT_DEPS:
                subj.append(c.i)
            elif d in OBJECT_DEPS:
                obj.append(c.i)
            elif d in DATIVE_DEPS and c.pos_ != "ADP":
                dat.append(c.i)  # bare indirect object: "gave him a book"
            # a prepositional dative ("for him", "to her") falls through to
            # the obliques below and keeps its English order among them
            elif d in PARTICLE_DEPS:
                prt.append(c.i)
            elif d in VERB_COMPLEX_DEPS:
                vgroup.append(c.i)
            else:
                obl.append(c.i)
        core = (obl + dat + obj) if obliques_before_object else (dat + obj + obl)
        if auxiliaries_after_verb:
            order = front + subj + core + prt + [tok.i] + vgroup + end
        else:
            order = front + subj + core + vgroup + prt + [tok.i] + end
        new.layout[tok.i] = [x for i in order for x in [i] + glued.get(i, [])]
    new.applied.append("81A")
    return new


# --------------------------------------------------------------------------
# 85A  Order of Adposition and Noun Phrase  ->  Postpositions
# --------------------------------------------------------------------------

ADPOSITION_DEPS = {"prep", "agent", "dative"}
ADPOSITION_OBJECT_DEPS = {"pobj", "pcomp"}


def is_adposition(tok: Token) -> bool:
    """True for a preposition heading a phrase. spaCy labels these prep or
    agent ("by"), and sometimes dative ("for him", "to her")."""
    return tok.dep_ in {"prep", "agent"} or (tok.dep_ == "dative" and tok.pos_ == "ADP")

# "a cup of tea", "the rest of them": "of" here is not a possessor.
QUANTITY_NOUNS = {
    "cup", "glass", "bottle", "bowl", "piece", "slice", "bit", "lot", "kind",
    "sort", "type", "number", "group", "part", "couple", "pair", "bunch",
    "rest", "amount", "plenty", "handful", "majority", "most", "some", "all",
    "half", "dozen", "pile", "series", "variety", "pinch", "drop",
}


def is_of_genitive(tok: Token) -> bool:
    """True if ``tok`` is the "of" of a possessive "X of Y" (both nouns).

    Shared by 85A and 86A so that the two functions agree on which "of"
    phrases are genitives."""
    if tok.lower_ != "of" or tok.dep_ != "prep":
        return False
    head = tok.head
    if head.pos_ not in {"NOUN", "PROPN"} or head.lemma_.lower() in QUANTITY_NOUNS:
        return False
    return any(c.dep_ == "pobj" and c.pos_ in NOUN_POS for c in tok.children)


def postpositions_85A(state: WordOrderState, *,
                      adnominal_pp_before_noun: bool = True) -> WordOrderState:
    """85A: turn prepositions into postpositions.

    Input
        state : WordOrderState
            Needs ``dep_`` (prep / agent and their pobj / pcomp) and ``pos_``
            of the word a phrase attaches to.
        adnominal_pp_before_noun : bool, default True
            A prepositional phrase that modifies a noun ("the bone in the
            house") also moves before that noun, like other noun modifiers in a
            head-final language: "house in bone". "of"-genitives are left to
            86A (see is_of_genitive).

    Output
        New WordOrderState where every adposition comes after its object:
            "in the house" -> "the house in", "by the dog" -> "the dog by".
        Adpositions with no object (stranded, "the house he lives in") and
        phrasal-verb particles (dep prt, e.g. "pick up") are not changed.

    spaCy labels used
        prep, agent, dative with pos_ ADP (adposition); pobj, pcomp (its
        object); head.pos_ NOUN/PROPN.

    Example
        "She put the cup on the table." -> "She put the cup the table on."
    """
    new = state.copy()
    for tok in state.doc:
        if not is_adposition(tok):
            continue
        objs = [c.i for c in tok.children if c.dep_ in ADPOSITION_OBJECT_DEPS]
        if not objs:
            continue  # stranded preposition
        others = [i for i in new.layout[tok.i] if i != tok.i and i not in objs]
        before = [i for i in others if i < tok.i]
        after = [i for i in others if i > tok.i]
        new.layout[tok.i] = before + objs + [tok.i] + after
        if (adnominal_pp_before_noun and tok.head.pos_ in {"NOUN", "PROPN"}
                and not is_of_genitive(tok)):
            new._place(tok.head.i, tok.i, "before", RANK_ADNOMINAL_PP)
    new.applied.append("85A")
    return new


# --------------------------------------------------------------------------
# 86A  Order of Genitive and Noun  ->  Genitive-Noun
# --------------------------------------------------------------------------

def genitive_noun_86A(state: WordOrderState, *, drop_of: bool = True,
                      drop_possessive_marker: bool = False) -> WordOrderState:
    """86A: put every possessor before the possessed noun.

    Input
        state : WordOrderState
            Needs ``dep_`` (poss, prep + pobj, case), ``lower_`` / ``lemma_``
            and ``pos_``.
        drop_of : bool, default True
            Remove "of" from "X of Y" genitives. It stays in ``state.dropped``
            with its tags, so Group 1 can turn it into a genitive case ending.
        drop_possessive_marker : bool, default False
            Also remove the "'s" of "John's". Kept by default because Group 1
            may mark genitives with an affix (49A Number of Cases).

    Output
        New WordOrderState where:
            "the roof of the house" -> "the house the roof"   (drop_of=True)
            "John's dog"            -> unchanged (already Genitive-Noun)
        The possessor goes first in its noun phrase, before relative clauses,
        articles and other pre-noun words. Quantity/partitive "of" ("a cup of
        tea", "one of them") is not treated as a genitive.

    spaCy labels used
        poss (possessor), prep "of" + pobj (of-genitive), case ("'s").

    Example
        "The roof of the house is red." -> "the house The roof is red."
    """
    new = state.copy()
    for tok in state.doc:
        if is_of_genitive(tok):
            new._place(tok.head.i, tok.i, "before", RANK_GENITIVE)
            if drop_of:
                new.dropped.add(tok.i)
        elif tok.dep_ == "poss":
            new._place(tok.head.i, tok.i, "before", RANK_GENITIVE)
        elif tok.dep_ == "case" and drop_possessive_marker:
            new.dropped.add(tok.i)
    new.applied.append("86A")
    return new


# --------------------------------------------------------------------------
# 87A  Order of Adjective and Noun  ->  Noun-Adjective
# --------------------------------------------------------------------------

def noun_adjective_87A(state: WordOrderState) -> WordOrderState:
    """87A: put attributive adjectives after the noun.

    Input
        state : WordOrderState
            Needs ``dep_`` == "amod" and the head's ``pos_``.

    Output
        New WordOrderState where every amod phrase directly follows its noun,
        keeping the English order of several adjectives and anything attached
        to the adjective ("very big", "big and brown"):
            "a big brown dog" -> "a dog big brown"
        Predicate adjectives ("the dog is big") are clause complements and are
        handled by 81A. Noun-noun compounds (dep compound) are not adjectives
        and stay before the noun.

    spaCy labels used
        amod; head.pos_ in NOUN / PROPN / PRON.

    Example
        "The old man saw a small bird." -> "The man old saw a bird small."
    """
    new = state.copy()
    for tok in state.doc:
        if tok.dep_ == "amod" and tok.head.pos_ in NOUN_POS and tok.i < tok.head.i:
            # adjectives already after the noun ("the sailor, tired and
            # happy, ...") stay where they are, with their commas
            new._place(tok.head.i, tok.i, "after", RANK_ADJECTIVE)
    new.applied.append("87A")
    return new


# --------------------------------------------------------------------------
# 90A  Order of Relative Clause and Noun  ->  Relative clause-Noun
# --------------------------------------------------------------------------

RELATIVIZER_TAGS = {"WDT", "WP"}  # which, that, who, whom (not "whose" = WP$)


def relcl_noun_90A(state: WordOrderState, *,
                   drop_relativizer: bool = True) -> WordOrderState:
    """90A: put relative clauses before the noun they modify.

    Input
        state : WordOrderState
            Needs ``dep_`` == "relcl", and ``tag_`` / ``dep_`` of the words
            directly under the relative clause's verb.
        drop_relativizer : bool, default True
            Remove the relative pronoun (who / whom / which / that) when it is a
            direct dependent of the relative clause verb. Prenominal relative
            clauses usually have no relative pronoun.

    Output
        New WordOrderState where every relcl phrase is placed before its noun
        (after a genitive, before articles):
            "the man who saw the dog" -> "saw the dog the man"
        With 81A also applied the clause itself becomes SOV:
            -> "the dog saw the man".
        For a non-restrictive clause the commas around it are dropped too:
            "The dog, which had brown fur, loved ..." -> "had brown fur The dog loved ..."
        Kept as is: "whose", relative adverbs (where / when), and pronouns
        inside a preposition ("in which") - see README edge cases.

    spaCy labels used
        relcl; tag_ WDT / WP for relativizers.

    Example
        "I know the man who owns the dog." -> "I know owns the dog the man."
    """
    new = state.copy()
    for tok in state.doc:
        if tok.dep_ != "relcl":
            continue
        new._place(tok.head.i, tok.i, "before", RANK_RELCL)
        if drop_relativizer:
            for c in tok.children:
                if c.tag_ in RELATIVIZER_TAGS:
                    new.dropped.add(c.i)
        # Non-restrictive clause ("The dog, which had brown fur, ..."): the
        # commas around it have nothing to separate once it is before the noun.
        doc = state.doc
        for i in (tok.left_edge.i - 1, tok.right_edge.i, tok.right_edge.i + 1):
            if 0 <= i < len(doc) and doc[i].text == "," and doc[i].head.i in (tok.i, tok.head.i):
                new.dropped.add(i)
    new.applied.append("90A")
    return new


# --------------------------------------------------------------------------
# Pipeline wrapper
# --------------------------------------------------------------------------

FEATURES: dict[str, Callable[[WordOrderState], WordOrderState]] = {
    "81A": sov_81A,
    "85A": postpositions_85A,
    "86A": genitive_noun_86A,
    "87A": noun_adjective_87A,
    "90A": relcl_noun_90A,
}


def apply_word_order(doc: Doc, features: Iterable[str] = tuple(FEATURES)) -> WordOrderState:
    """Run the chosen word order functions on one parsed Doc.

    Input
        doc : spacy.tokens.Doc   (from nlp(text))
        features : WALS IDs to apply, e.g. ("81A", "87A"); default all five.
    Output
        WordOrderState; use .tokens() for the next pipeline step, .text() to read it.
    """
    state = WordOrderState.from_doc(doc)
    for fid in features:
        state = FEATURES[fid](state)
    return state
