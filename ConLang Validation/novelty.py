"""Novelty: how different a profile is from every real language in WALS.

A. Closest language: the share of features on which the profile and a WALS
   language disagree, counting only features both have, and only languages that
   share at least min_overlap feature groups. Feature groups (see FEATURE_GROUPS)
   get a total weight of 1 so near-duplicate features are not counted several
   times, and sharing any feature of a group counts as sharing that group.
B. Percentile: every real language's distance to its own closest language is
   computed the same way (on the profile's features), and the profile is ranked
   among them. By default languages in the same genus are not allowed as each
   other's closest match, since a conlang has no relatives either.
C. Exact matches: languages that agree on every shared feature.
D. Soft distance: like A, but uses the profile's proportions instead of only its
   dominant answer, so 60% AdjN is a partial match for a Noun-Adjective language.
"""

from collections import Counter

import numpy as np

from wals_mapping import FEATURE_MAPS, feature_group

_reference_cache = {}


def group_weights(space, cols: list) -> np.ndarray:
    groups = [feature_group(space.features[j]) for j in cols]
    sizes = Counter(groups)
    return np.array([1.0 / sizes[g] for g in groups])


def group_matrix(space, cols: list) -> np.ndarray:
    """cols x groups indicator, for counting how many groups a language shares."""
    groups = list(dict.fromkeys(feature_group(space.features[j]) for j in cols))
    return np.array([[feature_group(space.features[j]) == g for g in groups] for j in cols], dtype=int)


def _weighted_mean(values: np.ndarray, shared: np.ndarray, weights: np.ndarray) -> np.ndarray:
    wsum = shared @ weights
    total = np.where(shared, values, 0.0) @ weights
    return np.where(wsum > 0, total / np.where(wsum > 0, wsum, 1.0), np.nan)


def soft_tables(space, profile, cols: list) -> list:
    """For each column: the distance from the profile to each WALS code of that feature."""
    tables = []
    for j in cols:
        f = space.features[j]
        answer, fmap = profile.answers[f], FEATURE_MAPS[f]
        codes = space.codes[f]
        p = np.array([answer.proportions.get(c, 0.0) for c in codes])
        table = 1.0 - p  # total variation distance to a single answer
        if fmap.mixed_code:
            # "No dominant order" is compared with an even split over the orders.
            order_codes = set(fmap.categories.values())
            uniform = np.array([1.0 / len(order_codes) if c in order_codes else 0.0 for c in codes])
            table[space.code_index[f][fmap.mixed_code]] = 0.5 * np.abs(p - uniform).sum()
        tables.append(table)
    return tables


def reference_distances(space, cols: list, min_overlap: int, exclude_same_genus: bool) -> np.ndarray:
    """Each real language's distance to its closest other language, on the given features."""
    key = (tuple(cols), min_overlap, exclude_same_genus)
    if key in _reference_cache:
        return _reference_cache[key]

    weights, groups = group_weights(space, cols), group_matrix(space, cols)
    W = space.matrix[:, cols]
    keep = (((W >= 0) @ groups) > 0).sum(1) >= min_overlap
    W, genus = W[keep], space.genus[keep]
    has = W >= 0

    shared = has[:, None, :] & has[None, :, :]
    mismatch = (W[:, None, :] != W[None, :, :]).astype(float)
    dist = _weighted_mean(mismatch, shared, weights)

    shared_groups = ((shared @ groups) > 0).sum(2)
    invalid = (shared_groups < min_overlap) | np.eye(len(W), dtype=bool)
    if exclude_same_genus:
        invalid |= genus[:, None] == genus[None, :]
    dist[invalid] = np.inf
    nearest = dist.min(1)
    _reference_cache[key] = nearest[np.isfinite(nearest)]
    return _reference_cache[key]


def novelty(space, profile, min_overlap: int = 3, exclude_same_genus: bool = True) -> dict:
    cols = [space.col[f] for f in profile.features if f in space.col]
    weights, groups = group_weights(space, cols), group_matrix(space, cols)
    if groups.shape[1] < min_overlap:
        return {"error": f"only {groups.shape[1]} feature groups with enough data (need {min_overlap})"}

    vec = space.encode(profile)[cols]
    W = space.matrix[:, cols]
    shared = W >= 0
    overlap = shared.sum(1)
    candidates = ((shared @ groups) > 0).sum(1) >= min_overlap
    if not candidates.any():
        return {"error": f"no WALS language shares {min_overlap}+ of these feature groups"}

    hard = _weighted_mean((W != vec).astype(float), shared, weights)
    tables = soft_tables(space, profile, cols)
    soft_values = np.stack([tables[c][np.maximum(W[:, c], 0)] for c in range(len(cols))], axis=1)
    soft = _weighted_mean(soft_values, shared, weights)

    def closest(dist):
        # Smallest distance; ties go to the language sharing the most features.
        order = np.lexsort((-overlap, dist))
        return next(i for i in order if candidates[i])

    i = closest(np.where(candidates, hard, np.inf))
    differences = [
        {"feature": space.features[cols[c]],
         "profile": space.code_name(space.codes[space.features[cols[c]]][vec[c]]),
         "language": space.code_name(space.codes[space.features[cols[c]]][W[i, c]])}
        for c in range(len(cols)) if W[i, c] >= 0 and W[i, c] != vec[c]
    ]
    s = closest(np.where(candidates, soft, np.inf))

    exact = candidates & np.isclose(hard, 0.0)
    reference = reference_distances(space, cols, min_overlap, exclude_same_genus)
    d = hard[i]
    percentile = 100.0 * (np.sum(reference < d - 1e-9) + 0.5 * np.sum(np.isclose(reference, d))) / len(reference)

    return {
        "features": [space.features[j] for j in cols],
        "n_groups": int(groups.shape[1]),
        "n_candidates": int(candidates.sum()),
        "closest": {"language": space.language_name(i), "distance": float(d),
                    "shared_features": int(overlap[i]), "differences": differences},
        "soft_closest": {"language": space.language_name(s), "distance": float(soft[s]),
                         "shared_features": int(overlap[s])},
        "exact_matches": int(exact.sum()),
        "exact_matches_all_features": int((exact & (overlap == len(cols))).sum()),
        "percentile": float(percentile),
        "n_reference": int(len(reference)),
        "exclude_same_genus": exclude_same_genus,
    }
