"""Novelty and plausibility of the conlang, measured against WALS (wals_data/).

Comparison set (the same for every measure)
  WALS languages with values for at least MIN_FEATURES features that also have
  a nearest neighbour under the novelty rule below.

Novelty
  Similarity of two languages = share of the features both have values for on
  which they agree, counting only features the conlang has a value for (also
  when comparing two WALS languages). Only languages with at least
  MIN_FEATURES features are compared, and only pairs sharing at least
  MIN_OVERLAP of the conlang's features.
  Reports the conlang's closest WALS languages, and where its nearest-neighbour
  similarity falls among the comparison set's nearest-neighbour similarities.

Plausibility (each language is scored against all *other* languages)
  Like novelty, every language is scored only on the conlang's features.
  Unattested pairs: pairs of a language's values that no other language has
  together (only pairs where another language has values for both features).
  Languages are compared on the share of their testable pairs that are unattested.
  Likelihood: for each language, the mean over its (conlang) features of
  log((other languages with that value + 1) /
      (other languages with a value for that feature + number of possible values)).
  The +1 keeps values unique to one language from giving log(0).
  Every WALS language is used as background for these counts; the conlang is
  compared only against the comparison set.

Usage: python validate_conlang.py
"""
import csv
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "wals_data"
CONLANG = "conlang"
MIN_FEATURES = 50  # a WALS language needs this many features (of any kind) to be compared
MIN_OVERLAP = 20   # two languages need this many of the conlang's features in common to be compared
TOP = 5
BINS = 20  # histogram bins per panel in the figure


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

n_feats = P.sum(1)
eligible = wals & (n_feats >= MIN_FEATURES)

# ---- Novelty (all similarities use only the conlang's features)
Pk, Xk = P[:, con_feats], X[:, np.isin(code_feat, con_feats)]
shared = Pk @ Pk.T
pool = eligible[:, None] & eligible[None, :]
pool[c] = pool[:, c] = eligible
with np.errstate(invalid="ignore", divide="ignore"):
    sim = np.where(pool & (shared >= MIN_OVERLAP), (Xk @ Xk.T) / shared, np.nan)
np.fill_diagonal(sim, np.nan)

# Comparison set used by every measure: eligible languages with a WALS nearest neighbour
ref = eligible & ~np.isnan(sim[:, wals]).all(axis=1)

con_sim = sim[c]
closest = [i for i in np.argsort(-np.nan_to_num(con_sim, nan=-1))[:TOP] if not np.isnan(con_sim[i])]
con_nn = con_sim[closest[0]]

nn = np.nanmax(sim[np.ix_(ref, wals)], axis=1)
nn_pct = 100 * (nn < con_nn).mean()

# ---- Plausibility: unattested value pairs (a pair is unattested for a language
#      if it is the only language with both values, but not the only one with both features)
pair_n = X.T @ X                                    # languages with both values
feat_n = (P.T @ P)[np.ix_(code_feat, code_feat)]    # languages with both features
testable = (feat_n >= 2) & (code_feat[:, None] != code_feat[None, :])
lone = testable & (pair_n == 1)
con_code = np.isin(code_feat, con_feats)          # values of the conlang's features
lone_k, testable_k = lone[np.ix_(con_code, con_code)], testable[np.ix_(con_code, con_code)]
n_lone = ((Xk @ lone_k) * Xk).sum(1) / 2
lone_rate = n_lone / np.maximum(((Xk @ testable_k) * Xk).sum(1) / 2, 1)
pair_pct = 100 * (lone_rate[ref] > lone_rate[c]).mean()
unattested = sorted(
    ((int(feat_n[ci[a], ci[b]]) - 1, a, b)
     for a, b in combinations(con_codes, 2) if lone[ci[a], ci[b]]),
    reverse=True,
)

# ---- Plausibility: mean log-probability of each language's values
n_values = Counter(r["Parameter_ID"] for r in read("codes.csv"))
k = np.array([n_values[f] for f in feats])[code_feat]
logp = np.log(X.sum(0) / (P.sum(0)[code_feat] - 1 + k))  # (n_value - 1 + 1) / (n_feat - 1 + k)
score = (Xk @ logp[con_code]) / np.maximum(Pk.sum(1), 1)
rank = 1 + int((score[ref] > score[c]).sum())
lik_pct = 100 * (score[ref] < score[c]).mean()
rarest = sorted(con_codes, key=lambda k: logp[ci[k]])[:TOP]

# ---- Report
print(f"Conlang: {int(n_feats[c])} WALS features")
print(f"Comparison set: {int(ref.sum())} WALS languages with >= {MIN_FEATURES} features and a neighbour "
      f"sharing >= {MIN_OVERLAP} of the conlang's features ({int((eligible & ~ref).sum())} dropped)\n")

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
      f"WALS median {100 * np.median(lone_rate[ref]):.2f}% "
      f"({100 * (n_lone[ref] > 0).mean():.0f}% of WALS languages have at least one)")
print(f"  Percentile: {pair_pct:.1f} -> fewer unattested pairs than {pair_pct:.1f}% of languages "
      f"(higher = more plausible)")
print(f"\n  Mean log-likelihood: conlang {score[c]:.3f}, WALS median {np.median(score[ref]):.3f}")
print(f"  Rank: {rank} of {int(ref.sum()) + 1} (1 = most typical); "
      f"more likely than {lik_pct:.1f}% of languages")
print(f"  Rarest conlang values:")
for k in rarest:
    print(f"    {label(k):<70} p = {np.exp(logp[ci[k]]):.3f}")

# ---- Figure: where the conlang falls in each WALS distribution
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

INK, MUTED, BAR, MARK = "#0b0b0b", "#52514e", "#9ec5f4", "#eb6834"
panels = [
    ("Novelty", 100 * nn, 100 * con_nn, "Similarity to closest language (%)",
     f"Percentile {nn_pct:.1f} (lower = more novel)"),
    ("Plausibility: unattested pairs", 100 * lone_rate[ref], 100 * lone_rate[c],
     "Unattested value pairs (%)", f"Percentile {pair_pct:.1f} (higher = more plausible)"),
    ("Plausibility: likelihood", score[ref], score[c], "Mean log-likelihood",
     f"Percentile {lik_pct:.1f} (higher = more typical)"),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
XMAX = {1: 12}  # panel index -> x-axis limit; values above it are counted in the last bar
for i, (ax, (title, dist, con, xlabel, note)) in enumerate(zip(axes, panels)):
    if i in XMAX:
        dist = np.minimum(dist, XMAX[i])
        bins = np.linspace(0, XMAX[i], BINS + 1)
    else:
        bins = BINS
    ax.hist(dist, bins=bins, color=BAR, edgecolor="white", linewidth=0.8, label="WALS languages")
    if i in XMAX:
        ax.set_xlim(0, XMAX[i])
    ax.axvline(con, color=MARK, linewidth=2, label="Conlang")
    ax.set_title(f"{title} (n={len(dist)} WALS languages)\n{note}", color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel("Languages", color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
out = Path(__file__).parent / "conlang_percentiles.png"
fig.savefig(out, dpi=200)
print(f"\nFigure saved to {out}")
