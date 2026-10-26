"""Plausibility: how typical a profile's answers are, alone and in combination.

E. Feature log probability: log P(answer) for each feature, where P is how often
   real languages have that answer. Reported as a sum and a mean (the mean is
   comparable across profiles with different numbers of features).
F. Pair log probability: for each ordered pair of features (A, B), log P(B's answer
   given A's answer), averaged over pairs. Pairs within a feature group are
   skipped, since those features nearly determine each other. Pairs with
   P < rare_threshold are listed as rare combinations.

Both are weighted by feature group (see FEATURE_GROUPS): each group has a total
weight of 1, so the four clause-order features count as one feature in E, and a
pair's weight is the product of its two features' weights in F.

All probabilities count genera (groups of closely related languages) rather than
languages, so large, well-documented families do not dominate: every genus with
data has a total weight of 1, split among its languages. A small count (alpha) is
added to every answer so unattested answers get a low, finite probability.
Logs are natural logs.
"""

import math
from collections import Counter

import numpy as np

from wals_mapping import feature_group


def genus_weights(space, mask: np.ndarray) -> np.ndarray:
    counts = np.bincount(space.genus[mask], minlength=space.genus.max() + 1)
    weights = np.zeros(len(mask))
    weights[mask] = 1.0 / counts[space.genus[mask]]
    return weights


def answer_probabilities(space, feature: str, alpha: float) -> np.ndarray:
    col = space.matrix[:, space.col[feature]]
    mask = col >= 0
    n_codes = len(space.codes[feature])
    counts = np.bincount(col[mask], weights=genus_weights(space, mask)[mask], minlength=n_codes)
    return (counts + alpha) / (counts.sum() + alpha * n_codes)


def conditional_probability(space, a: str, code_a: int, b: str, code_b: int, alpha: float):
    """P(b = code_b | a = code_a), plus raw language counts behind it."""
    A, B = space.matrix[:, space.col[a]], space.matrix[:, space.col[b]]
    both = (A >= 0) & (B >= 0)
    given = both & (A == code_a)
    n_codes = len(space.codes[b])
    counts = np.bincount(B[given], weights=genus_weights(space, both)[given], minlength=n_codes)
    p = (counts[code_b] + alpha) / (counts.sum() + alpha * n_codes)
    return float(p), int(given.sum()), int((given & (B == code_b)).sum())


def plausibility(space, profile, alpha: float = 0.5, rare_threshold: float = 0.05) -> dict:
    features = [f for f in profile.features if f in space.col]
    if not features:
        return {"error": "no features with enough data"}
    vec = space.encode(profile)
    group_sizes = Counter(feature_group(f) for f in features)
    weight = {f: 1.0 / group_sizes[feature_group(f)] for f in features}

    per_feature = []
    for f in features:
        k = vec[space.col[f]]
        p = float(answer_probabilities(space, f, alpha)[k])
        per_feature.append({"feature": f, "answer": space.code_name(space.codes[f][k]),
                            "prob": p, "logprob": math.log(p), "weight": weight[f]})
    e_sum = sum(x["weight"] * x["logprob"] for x in per_feature)

    pairs = []
    for a in features:
        for b in features:
            if a == b or feature_group(a) == feature_group(b):
                continue
            ka, kb = vec[space.col[a]], vec[space.col[b]]
            p, n_given, n_both = conditional_probability(space, a, ka, b, kb, alpha)
            pairs.append({
                "given": a, "given_answer": space.code_name(space.codes[a][ka]),
                "feature": b, "answer": space.code_name(space.codes[b][kb]),
                "prob": p, "logprob": math.log(p), "weight": weight[a] * weight[b],
                "n_languages_given": n_given, "n_languages_both": n_both,
            })
    pair_weight = sum(x["weight"] for x in pairs)

    return {
        "features": features,
        "n_groups": len(group_sizes),
        "feature_logprob_sum": e_sum,
        "feature_logprob_mean": e_sum / len(group_sizes),
        "per_feature": per_feature,
        "pair_logprob_mean": (sum(x["weight"] * x["logprob"] for x in pairs) / pair_weight) if pairs else None,
        "n_pairs": len(pairs),
        "rare_pairs": sorted((x for x in pairs if x["prob"] < rare_threshold), key=lambda x: x["prob"]),
        "pairs": pairs,
        "alpha": alpha,
        "rare_threshold": rare_threshold,
    }
