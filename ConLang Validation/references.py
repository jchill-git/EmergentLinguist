"""Reference cases for interpreting novelty and plausibility scores.

- English: the untranslated sentences. English is in WALS, so expect novelty near
  0 and high plausibility.
- Translator seeds: the same translator with several seeds, to show the spread.
- Random: each feature's answer picked uniformly at random from the answers our
  tests can produce. Expect high novelty and low plausibility.
"""

import random
import statistics

from novelty import novelty
from plausibility import plausibility
from wals_mapping import FEATURE_MAPS, Answer, Profile

HEADLINE_KEYS = ["distance", "percentile", "exact_matches", "feature_logprob_mean",
                 "pair_logprob_mean", "n_rare_pairs"]


def score_profile(space, profile, cfg) -> dict:
    return {
        "novelty": novelty(space, profile, cfg.min_overlap, not cfg.include_same_genus),
        "plausibility": plausibility(space, profile, cfg.alpha, cfg.rare_threshold),
    }


def headline(scores: dict) -> dict:
    """The few numbers shown in the comparison table (None where not computable)."""
    nov, pla = scores["novelty"], scores["plausibility"]
    has_nov, has_pla = "error" not in nov, "error" not in pla
    return {
        "distance": nov["closest"]["distance"] if has_nov else None,
        "percentile": nov["percentile"] if has_nov else None,
        "exact_matches": nov["exact_matches"] if has_nov else None,
        "feature_logprob_mean": pla["feature_logprob_mean"] if has_pla else None,
        "pair_logprob_mean": pla["pair_logprob_mean"] if has_pla else None,
        "n_rare_pairs": len(pla["rare_pairs"]) if has_pla else None,
    }


def summarize(headlines: list) -> dict:
    """Mean, standard deviation, min and max of each headline number."""
    summary = {}
    for key in HEADLINE_KEYS:
        values = [h[key] for h in headlines if h[key] is not None]
        if values:
            summary[key] = {
                "mean": statistics.fmean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values), "max": max(values), "n": len(values),
            }
        else:
            summary[key] = None
    return summary


def random_profiles(features: list, n: int, seed: int = 0) -> list:
    rng = random.Random(seed)
    profiles = []
    for _ in range(n):
        profile = Profile()
        for f in features:
            fmap = FEATURE_MAPS[f]
            code = rng.choice(fmap.producible_codes())
            if code == fmap.mixed_code:
                orders = list(dict.fromkeys(fmap.categories.values()))
                proportions = {c: 1.0 / len(orders) for c in orders}
            else:
                proportions = {code: 1.0}
            profile.answers[f] = Answer(code, proportions, n=0)
        profiles.append(profile)
    return profiles
