"""Word order and related tests written for this suite (not from ud_llm).

Each test looks at the UD annotation that travels with every word, so it works on
both the English sentences and their conlang translations. Tests that need to
recognise specific words (e.g. "very") use the original English lemma, stored by
annotation.py in each token's misc field as EnLemma.

Like the ud_llm tests, each returns a distribution of observations and the most
frequent answer. Deciding whether there is enough data, and whether an answer is
dominant, happens later in wals_mapping.py.

  91A   Order of Degree Word and Adjective         "very old" vs. "old very"
  92A   Position of Polar Question Particles       where a yes/no question word goes
  93A   Position of Interrogative Phrases          "What did he eat?" vs. "He ate what?"
  112A  Negative Morphemes                         what kind of word/part "not" is
  116A  Polar Questions                            how yes/no questions are marked
  143A  Order of Negative Morpheme and Verb        "not leave" vs. "leave not"
"""

from feature_tests import FeatureTest, TestResult, register_test

# Relations that start a new clause (same list the ud_llm scripts use).
CLAUSE_RELATIONS = {"root", "conj", "parataxis", "csubj", "ccomp", "xcomp", "advcl", "acl"}
# Dependents that belong to the clause rather than to a phrase inside it.
CLAUSE_LEVEL_RELATIONS = CLAUSE_RELATIONS | {
    "nsubj", "obj", "iobj", "obl", "aux", "cop", "mark", "punct", "cc", "expl",
    "discourse", "vocative", "dislocated",
}
DEGREE_WORDS = {
    "very", "quite", "too", "so", "rather", "extremely", "really", "pretty", "fairly",
    "somewhat", "more", "most", "less", "least", "highly", "incredibly", "terribly",
    "slightly", "almost", "completely", "totally", "truly", "especially",
}
NEGATIVE_WORD_UPOS = {"PART", "ADV", "AUX"}


# --- helpers ----------------------------------------------------------------

def base(deprel: str) -> str:
    return (deprel or "").split(":")[0]


def feat(tok, name: str):
    return (tok["feats"] or {}).get(name)


def english_lemma(tok) -> str:
    return ((tok["misc"] or {}).get("EnLemma") or tok["lemma"] or "").lower()


class Sentence:
    """Index of one sentence's words (multi-word tokens are ignored)."""

    def __init__(self, tokenlist):
        self.words = [t for t in tokenlist if isinstance(t["id"], int)]
        self.by_id = {t["id"]: t for t in self.words}
        self.children = {}
        for t in self.words:
            self.children.setdefault(t["head"], []).append(t)
        self.root = next((t for t in self.words if t["head"] == 0), None)

    def kids(self, tok, *relations):
        return [c for c in self.children.get(tok["id"], [])
                if not relations or base(c["deprel"]) in relations]

    def is_question(self) -> bool:
        return any(t["upos"] == "PUNCT" and t["form"] == "?" for t in self.words)

    def is_statement(self) -> bool:
        return any(t["upos"] == "PUNCT" and t["form"] == "." for t in self.words) and not self.is_question()

    def clause_head(self, tok):
        while base(tok["deprel"]) not in CLAUSE_RELATIONS and tok["head"] in self.by_id:
            tok = self.by_id[tok["head"]]
        return tok

    def clause_words(self, head):
        """Non-punctuation words whose clause is headed by head, in sentence order."""
        return [t for t in self.words
                if t["upos"] != "PUNCT" and self.clause_head(t)["id"] == head["id"]]

    def subtree(self, tok, skip_relations=()):
        ids = [tok["id"]]
        for c in self.kids(tok):
            if base(c["deprel"]) not in skip_relations:
                ids.extend(self.subtree(c))
        return ids

    def finite_verb(self, head):
        """The first auxiliary/copula of a clause, else its head if it is a verb."""
        helpers = sorted(self.kids(head, "aux", "cop"), key=lambda t: t["id"])
        if helpers:
            return helpers[0]
        return head if head["upos"] in {"VERB", "AUX"} else None

    def negative_words(self):
        return [t for t in self.words
                if t["upos"] in NEGATIVE_WORD_UPOS and feat(t, "Polarity") == "Neg"]

    def interrogative_words(self):
        """Question words like who/what, excluding question particles."""
        return [t for t in self.words if feat(t, "PronType") == "Int" and t["upos"] != "PART"]

    def question_particles(self):
        return [t for t in self.words if feat(t, "PronType") == "Int" and t["upos"] == "PART"]


class CountingTest(FeatureTest):
    """Counts one category per observation across all sentences.

    Subclasses set test_id, name and categories, and implement observe(), which
    returns the categories observed in one sentence (possibly none).
    """

    categories = []

    def prepare(self, sentences: list) -> None:
        """Optional pass over all sentences before observe() is called."""

    def observe(self, sentence: Sentence) -> list:
        raise NotImplementedError

    def run(self, sentences: list) -> TestResult:
        indexed = [Sentence(s) for s in sentences]
        self.prepare(indexed)
        counts = {c: 0 for c in self.categories}
        for s in indexed:
            for category in self.observe(s):
                counts[category] += 1
        final = max(counts, key=counts.get) if any(counts.values()) else "No valid observations"
        return TestResult(self.test_id, self.name, final, counts)


# --- tests ------------------------------------------------------------------

@register_test
class DegreeWordAdjective(CountingTest):
    """A degree word ("very", "too", "more", ...) attached to an adjective."""

    test_id = "91A"
    name = "Order of Degree Word and Adjective"
    categories = ["DegAdj", "AdjDeg"]

    def observe(self, s):
        found = []
        for adj in s.words:
            if adj["upos"] != "ADJ":
                continue
            for dep in s.kids(adj, "advmod"):
                if english_lemma(dep) in DEGREE_WORDS:
                    found.append("DegAdj" if dep["id"] < adj["id"] else "AdjDeg")
        return found


@register_test
class PolarQuestionParticle(CountingTest):
    """Yes/no questions: is there a question particle, and where in the clause is it?

    A question particle is a PART with PronType=Int (as in UD for e.g. Polish
    "czy" or Japanese "ka"). Yes/no questions are sentences ending in "?" with
    no question word like who/what.
    """

    test_id = "92A"
    name = "Position of Polar Question Particles"
    categories = ["Initial", "Final", "Second position", "Other position", "No question particle"]

    def observe(self, s):
        if not s.is_question() or s.interrogative_words() or s.root is None:
            return []
        particles = s.question_particles()
        if not particles:
            return ["No question particle"]
        clause = [t["id"] for t in s.clause_words(s.clause_head(particles[0]))]
        position = clause.index(particles[0]["id"])
        if position == 0:
            return ["Initial"]
        if position == len(clause) - 1:
            return ["Final"]
        return ["Second position" if position == 1 else "Other position"]


@register_test
class InterrogativePhrasePosition(CountingTest):
    """Content questions: does the question phrase ("what", "which man") start the clause?

    Only direct questions are counted: sentences ending in "?" whose question
    word belongs to the main clause.
    """

    test_id = "93A"
    name = "Position of Interrogative Phrases in Content Questions"
    categories = ["Initial", "Not initial"]

    def observe(self, s):
        if not s.is_question() or s.root is None:
            return []
        for word in s.interrogative_words():
            clause_head = s.clause_head(word)
            if clause_head["id"] != s.root["id"]:
                continue
            # Climb from e.g. "which" to "which man", but not past the clause head.
            phrase = word
            while (phrase["id"] != clause_head["id"] and phrase["head"] in s.by_id
                   and s.by_id[phrase["head"]]["id"] != clause_head["id"]
                   and base(phrase["deprel"]) in {"det", "amod", "advmod", "nummod", "nmod"}):
                phrase = s.by_id[phrase["head"]]
            skip = CLAUSE_LEVEL_RELATIONS if phrase["id"] == clause_head["id"] else ()
            start = min(s.subtree(phrase, skip_relations=skip))
            first = s.clause_words(clause_head)[0]["id"]
            return ["Initial" if start == first else "Not initial"]
        return []


@register_test
class NegativeMorphemes(CountingTest):
    """For each negated clause: is negation a separate particle, an auxiliary verb,
    part of the verb itself, or two negative words together?"""

    test_id = "112A"
    name = "Negative Morphemes"
    categories = ["Negative particle", "Negative auxiliary verb", "Negative affix", "Double negation"]

    def observe(self, s):
        negated = {}
        for neg in s.negative_words():
            if neg["head"] in s.by_id:
                negated.setdefault(neg["head"], []).append(neg)
        found = []
        for negs in negated.values():
            if len(negs) > 1:
                found.append("Double negation")
            elif negs[0]["upos"] == "AUX":
                found.append("Negative auxiliary verb")
            else:
                found.append("Negative particle")
        # Verbs marked negative with no separate negative word attached.
        negative_ids = {t["id"] for t in s.negative_words()}
        for tok in s.words:
            if (tok["upos"] in {"VERB", "AUX"} and feat(tok, "Polarity") == "Neg"
                    and tok["id"] not in negated and tok["id"] not in negative_ids):
                found.append("Negative affix")
        return found


@register_test
class PolarQuestions(CountingTest):
    """How yes/no questions differ from statements.

    - "Question particle": the question has a particle (see 92A).
    - "Interrogative word order": the subject and the finite verb are in the
      opposite order from the dominant order in statements ("Is he...?").
    - "Interrogative intonation only": neither; the question looks like a statement.
    Verb endings that mark questions cannot be seen in the annotation.
    """

    test_id = "116A"
    name = "Polar Questions"
    categories = ["Question particle", "Interrogative word order", "Interrogative intonation only"]

    def _subject_first(self, s):
        if s.root is None:
            return None
        subjects = s.kids(s.root, "nsubj", "csubj")
        verb = s.finite_verb(s.root)
        if not subjects or verb is None:
            return None
        return subjects[0]["id"] < verb["id"]

    def prepare(self, sentences):
        orders = [self._subject_first(s) for s in sentences if s.is_statement()]
        orders = [o for o in orders if o is not None]
        self.statement_subject_first = (sum(orders) >= len(orders) / 2) if orders else None

    def observe(self, s):
        if not s.is_question() or s.interrogative_words():
            return []
        if s.question_particles():
            return ["Question particle"]
        subject_first = self._subject_first(s)
        if subject_first is None or self.statement_subject_first is None:
            return []
        if subject_first != self.statement_subject_first:
            return ["Interrogative word order"]
        return ["Interrogative intonation only"]


@register_test
class NegativeVerbOrder(CountingTest):
    """Does the negative word come before or after the verb it negates?

    Only clauses with a full verb are counted (not "is/are" sentences like
    "Mary is not at home"), since WALS describes the position relative to the
    lexical verb. A clause with negative words on both sides counts as double
    negation.
    """

    test_id = "143A"
    name = "Order of Negative Morpheme and Verb"
    categories = ["NegV", "VNeg", "NegVNeg"]

    def observe(self, s):
        sides = {}
        for neg in s.negative_words():
            verb = s.by_id.get(neg["head"])
            if verb is None or verb["upos"] != "VERB":
                continue
            sides.setdefault(verb["id"], set()).add("before" if neg["id"] < verb["id"] else "after")
        return ["NegVNeg" if len(v) == 2 else ("NegV" if "before" in v else "VNeg")
                for v in sides.values()]
