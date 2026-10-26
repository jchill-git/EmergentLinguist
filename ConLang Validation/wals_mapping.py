"""Convert feature test results into WALS answers (a "profile").

Each test reports a distribution, e.g. {"AdjN": 12, "NAdj": 3}. This module maps
those categories onto WALS codes and decides the dominant answer:
  - an answer is dominant if it is at least DOMINANCE_RATIO times as frequent as
    the next one (the rule WALS uses; see "Determining Dominant Word Order");
  - otherwise the feature gets its "No dominant order" / "Mixed" code;
  - features with fewer than min_obs observations are left out of the profile.

To use a new test in novelty/plausibility scoring, add a FeatureMap for it below.
"""

from dataclasses import dataclass, field
from itertools import permutations

DOMINANCE_RATIO = 2.0


@dataclass
class FeatureMap:
    categories: dict  # test distribution key -> WALS code id (or a code WALS lacks)
    mixed_code: str = None  # code for "no dominant order"; None -> take the most frequent
    extra_names: dict = field(default_factory=dict)  # names for codes WALS does not have

    def producible_codes(self) -> list:
        codes = list(dict.fromkeys(self.categories.values()))
        return codes + [self.mixed_code] if self.mixed_code else codes


# 144A: WALS lists 15 of the 24 orders of S, O, V and Neg. The rest are unattested,
# so they get their own codes that no WALS language has.
_144A_ATTESTED = {
    "NegSVO": "144A-1", "SNegVO": "144A-2", "SVNegO": "144A-3", "SVONeg": "144A-4",
    "NegSOV": "144A-5", "SNegOV": "144A-6", "SONegV": "144A-7", "SOVNeg": "144A-8",
    "NegVSO": "144A-9", "VSNegO": "144A-10", "VSONeg": "144A-11", "NegVOS": "144A-12",
    "ONegVS": "144A-13", "OVNegS": "144A-14", "OSVNeg": "144A-15",
}
_144A_ORDERS = {"".join(p) for p in permutations(["S", "O", "V", "Neg"])}
_144A_UNATTESTED = {o: f"144A-{o}" for o in sorted(_144A_ORDERS - set(_144A_ATTESTED))}

FEATURE_MAPS = {
    "81A": FeatureMap(
        {"SOV": "81A-1", "SVO": "81A-2", "VSO": "81A-3", "VOS": "81A-4", "OVS": "81A-5", "OSV": "81A-6"},
        mixed_code="81A-7"),
    "82A": FeatureMap({"SV": "82A-1", "VS": "82A-2"}, mixed_code="82A-3"),
    "83A": FeatureMap({"OV": "83A-1", "VO": "83A-2"}, mixed_code="83A-3"),
    # WALS: "The sixth logical possibility, VXO, is not attested."
    "84A": FeatureMap(
        {"VOX": "84A-1", "XVO": "84A-2", "XOV": "84A-3", "OXV": "84A-4", "OVX": "84A-5", "VXO": "84A-VXO"},
        mixed_code="84A-6", extra_names={"84A-VXO": "VXO (unattested in WALS)"}),
    "85A": FeatureMap({"Postpositions": "85A-1", "Prepositions": "85A-2"}, mixed_code="85A-4"),
    "86A": FeatureMap({"GenN": "86A-1", "NGen": "86A-2"}, mixed_code="86A-3"),
    "87A": FeatureMap({"AdjN": "87A-1", "NAdj": "87A-2"}, mixed_code="87A-3"),
    "88A": FeatureMap({"DemN": "88A-1", "NDem": "88A-2"}, mixed_code="88A-6"),
    "89A": FeatureMap(
        {"NumN": "89A-1", "NNum": "89A-2", "Numeral only modifies verb": "89A-4"}, mixed_code="89A-3"),
    "90A": FeatureMap({"NRel": "90A-1", "RelN": "90A-2"}, mixed_code="90A-7"),
    "94A": FeatureMap({
        "Adverbial subordinators which appear at the beginning of the subordinate clause": "94A-1",
        "Adverbial subordinators which appear at the end of the subordinate clause": "94A-2",
        "Clause-internal adverbial subordinators": "94A-3",
    }, mixed_code="94A-5"),
    "144A": FeatureMap(
        {**_144A_ATTESTED, **_144A_UNATTESTED, "Morphological negation": "144A-20"},
        mixed_code="144A-16",
        extra_names={code: f"{order} (unattested in WALS)" for order, code in _144A_UNATTESTED.items()}),
    # Tests from word_order_tests.py.
    "91A": FeatureMap({"DegAdj": "91A-1", "AdjDeg": "91A-2"}, mixed_code="91A-3"),
    "92A": FeatureMap({
        "Initial": "92A-1", "Final": "92A-2", "Second position": "92A-3",
        "Other position": "92A-4", "No question particle": "92A-6",
    }, mixed_code="92A-5"),
    "93A": FeatureMap({"Initial": "93A-1", "Not initial": "93A-2"}, mixed_code="93A-3"),
    # 112A-5 is only for variation between a word and an affix, so no general mixed code.
    "112A": FeatureMap({
        "Negative affix": "112A-1", "Negative particle": "112A-2",
        "Negative auxiliary verb": "112A-3", "Double negation": "112A-6",
    }),
    "116A": FeatureMap({
        "Question particle": "116A-1", "Interrogative word order": "116A-4",
        "Interrogative intonation only": "116A-6",
    }),
    # 143A-6 is "Type 1 / Type 2", i.e. both NegV and VNeg are used.
    "143A": FeatureMap({"NegV": "143A-1", "VNeg": "143A-2", "NegVNeg": "143A-14"},
                       mixed_code="143A-6"),
    # WALS has no "mixed" value for 144B, so the most frequent position is used.
    "144B": FeatureMap({
        "At beginning of clause separated from verb": "144B-1",
        "Preceding verb, separated from verb, but not at beginning of clause": "144B-2",
        "Immediately preverbal": "144B-3",
        "Immediately postverbal": "144B-4",
        "Following verb, separated from verb, but not at end of clause": "144B-5",
        "At end of clause, separated from verb": "144B-6",
    }),
}

# Features that largely measure the same thing. Distances give each group a total
# weight of 1, and plausibility skips pairs within a group (they are near-certain
# given each other, e.g. 81A=SVO implies 82A=SV).
FEATURE_GROUPS = {
    "81A": "clause order", "82A": "clause order", "83A": "clause order", "84A": "clause order",
    "112A": "negation", "143A": "negation", "144A": "negation", "144B": "negation",
    "92A": "yes/no questions", "116A": "yes/no questions",
}


def feature_group(feature: str) -> str:
    return FEATURE_GROUPS.get(feature, feature)


def extra_codes() -> dict:
    return {f: list(m.categories.values()) for f, m in FEATURE_MAPS.items()}


def extra_code_names() -> dict:
    return {code: name for m in FEATURE_MAPS.values() for code, name in m.extra_names.items()}


@dataclass
class Answer:
    code: str  # the dominant WALS code (or the mixed code)
    proportions: dict  # WALS code -> share of observations
    n: int


@dataclass
class Profile:
    answers: dict = field(default_factory=dict)  # feature -> Answer
    skipped: dict = field(default_factory=dict)  # feature -> reason

    @property
    def features(self) -> list:
        return list(self.answers)


def to_answer(feature: str, distribution: dict, min_obs: int,
              dominance_ratio: float = DOMINANCE_RATIO):
    """Returns (Answer, None) or (None, reason it was skipped)."""
    fmap = FEATURE_MAPS[feature]
    counts = {}
    for key, value in distribution.items():
        if key in fmap.categories and value:
            code = fmap.categories[key]
            counts[code] = counts.get(code, 0) + value
    n = sum(counts.values())
    if n < min_obs:
        return None, f"{n} observation{'s' if n != 1 else ''} (need {min_obs})"

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_code, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    dominant = second == 0 or top >= dominance_ratio * second
    code = top_code if dominant or fmap.mixed_code is None else fmap.mixed_code
    return Answer(code, {c: v / n for c, v in counts.items()}, n), None


def profile_from_results(results: dict, min_obs: int,
                         dominance_ratio: float = DOMINANCE_RATIO) -> Profile:
    """results: test id -> TestResult (from feature_tests)."""
    profile = Profile()
    for test_id, result in results.items():
        if test_id not in FEATURE_MAPS:
            profile.skipped[test_id] = "no WALS mapping"
        elif result.error:
            profile.skipped[test_id] = f"test error: {result.error}"
        else:
            answer, reason = to_answer(test_id, result.distribution, min_obs, dominance_ratio)
            if answer:
                profile.answers[test_id] = answer
            else:
                profile.skipped[test_id] = reason
    return profile
