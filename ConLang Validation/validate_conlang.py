"""Novelty and plausibility of the conlang, measured against WALS (wals_data/).

Novelty
  Similarity of two languages = share of the features both have values for on
  which they agree, counting only features the conlang has a value for (also
  when comparing two WALS languages). Pairs sharing fewer than MIN_SHARED
  such features are ignored.
  Reports the conlang's closest WALS languages, and where its nearest-neighbour
  similarity falls among every WALS language's nearest-neighbour similarity.

Plausibility (each language is scored against all *other* languages)
  Unattested pairs: pairs of a language's values that no other language has
  together (only pairs where another language has values for both features).
  Languages are compared on the share of their testable pairs that are unattested.
  Likelihood: for each language, the mean over its features of
  log((other languages with that value + 1) /
      (other languages with a value for that feature + number of possible values)).
  The +1 keeps values unique to one language from giving log(0).
  Only languages with at least MIN_SHARED features are ranked.

Usage: python validate_conlang.py
"""
import csv
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "wals_data"
CONLANG = "conlang"
MIN_SHARED = 50
TOP = 5


def read(name):
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


# ---- Load WALS as matrices: P[lang, feature] = has value, X[lang, code] = has that value
lang_names = {r["ID"]: r["Name"] for r in read("languages.csv")}
code_names = {r["ID"]: r["Name"] for r in read("codes.csv")}
feat_names = {r["ID"]: r["Name"] for r in read("parameters.csv")}
rows = read("values.csv")

langs = sorted({r["Language_ID"] for r in rows})
feats = sorted({r["Parameter_ID"] for r in rows})
codes = sorted({r["Code_ID"] for r in rows})
li, fi, ci = ({k: i for i, k in enumerate(ks)} for ks in (langs, feats, codes))
code_feat = np.zeros(len(codes), dtype=int)

P = np.zeros((len(langs), len(feats)), dtype=np.float32)
X = np.zeros((len(langs), len(codes)), dtype=np.float32)
for r in rows:
    P[li[r["Language_ID"]], fi[r["Parameter_ID"]]] = 1
    X[li[r["Language_ID"]], ci[r["Code_ID"]]] = 1
    code_feat[ci[r["Code_ID"]]] = fi[r["Parameter_ID"]]

c = li[CONLANG]
wals = np.arange(len(langs)) != c


def label(code):
    feat = code.rsplit("-", 1)[0]
    return f"{feat} {feat_names[feat]} = {code_names[code]}"


con_codes = [codes[j] for j in np.flatnonzero(X[c])]
con_feats = [code_feat[ci[k]] for k in con_codes]

# ---- Novelty (all similarities use only the conlang's features)
Pk, Xk = P[:, con_feats], X[:, np.isin(code_feat, con_feats)]
shared = Pk @ Pk.T
with np.errstate(invalid="ignore", divide="ignore"):
    sim = np.where(shared >= MIN_SHARED, (Xk @ Xk.T) / shared, np.nan)
np.fill_diagonal(sim, np.nan)

con_sim = np.where(wals, sim[c], np.nan)
closest = [i for i in np.argsort(-np.nan_to_num(con_sim, nan=-1))[:TOP] if not np.isnan(con_sim[i])]
con_nn = con_sim[closest[0]]

sim_w = sim[np.ix_(wals, wals)]
has_nn = ~np.isnan(sim_w).all(axis=1)
nn = np.nanmax(sim_w[has_nn], axis=1)
nn_pct = 100 * (nn < con_nn).mean()

n_feats = P.sum(1)
ranked = n_feats >= MIN_SHARED

# ---- Plausibility: unattested value pairs (a pair is unattested for a language
#      if it is the only language with both values, but not the only one with both features)
pair_n = X.T @ X                                    # languages with both values
feat_n = (P.T @ P)[np.ix_(code_feat, code_feat)]    # languages with both features
testable = (feat_n >= 2) & (code_feat[:, None] != code_feat[None, :])
lone = testable & (pair_n == 1)
n_lone = ((X @ lone) * X).sum(1) / 2
lone_rate = n_lone / np.maximum(((X @ testable) * X).sum(1) / 2, 1)
pair_pct = 100 * (lone_rate[ranked] > lone_rate[c]).mean()
unattested = sorted(
    ((int(feat_n[ci[a], ci[b]]) - 1, a, b)
     for a, b in combinations(con_codes, 2) if lone[ci[a], ci[b]]),
    reverse=True,
)

# ---- Plausibility: mean log-probability of each language's values
n_values = Counter(r["Parameter_ID"] for r in read("codes.csv"))
k = np.array([n_values[f] for f in feats])[code_feat]
logp = np.log(X.sum(0) / (P.sum(0)[code_feat] - 1 + k))  # (n_value - 1 + 1) / (n_feat - 1 + k)
score = (X @ logp) / np.maximum(n_feats, 1)
rank = 1 + int((score[ranked] > score[c]).sum())
lik_pct = 100 * (score[ranked] < score[c]).mean()
rarest = sorted(con_codes, key=lambda k: logp[ci[k]])[:TOP]

# ---- Report
print(f"Conlang: {int(n_feats[c])} WALS features; comparisons need >= {MIN_SHARED} shared features\n")

print("NOVELTY")
print(f"  Closest WALS languages:")
for i in closest:
    print(f"    {lang_names[langs[i]]:<28} {100 * con_sim[i]:5.1f}% agree on {int(shared[c, i])} shared features")
print(f"  Nearest-neighbour similarity: conlang {100 * con_nn:.1f}%, "
      f"WALS median {100 * np.median(nn):.1f}% (n={len(nn)})")
print(f"  Percentile: {nn_pct:.1f} -> the conlang's nearest neighbour is closer than "
      f"{nn_pct:.1f}% of WALS languages' nearest neighbours (lower = more novel)\n")

print("PLAUSIBILITY")
print(f"  Value pairs no other language shares: {len(unattested)}")
for n, a, b in unattested:
    print(f"    {label(a)}  +  {label(b)}   (0 of {n} languages with both features)")
print(f"\n  Share of testable pairs unattested: conlang {100 * lone_rate[c]:.2f}%, "
      f"WALS median {100 * np.median(lone_rate[ranked & wals]):.2f}% "
      f"({100 * (n_lone[ranked & wals] > 0).mean():.0f}% of WALS languages have at least one)")
print(f"  Percentile: {pair_pct:.1f} -> fewer unattested pairs than {pair_pct:.1f}% of languages "
      f"(higher = more plausible)")
print(f"\n  Mean log-likelihood: conlang {score[c]:.3f}, WALS median {np.median(score[ranked & wals]):.3f}")
print(f"  Rank: {rank} of {int(ranked.sum())} (1 = most typical); "
      f"more likely than {lik_pct:.1f}% of languages")
print(f"  Rarest conlang values:")
for k in rarest:
    print(f"    {label(k):<70} p = {np.exp(logp[ci[k]]):.3f}")
