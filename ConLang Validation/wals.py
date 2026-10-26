"""Load the WALS CLDF tables and encode them as a language x feature matrix.

The tables are downloaded into wals_data/ on first use (or by running this file).
"""

import csv
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

WALS_REPO = "cldf-datasets/wals"
# Commit the bundled tables were taken from.
WALS_REF = "81298aa26d01fdc11b769bcc7fa87ad9e43d7ccd"
TABLES = ["values", "languages", "codes", "parameters"]
DATA_DIR = Path(__file__).resolve().parent / "wals_data"


def download(data_dir: Path = DATA_DIR, ref: str = WALS_REF, force: bool = False) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for table in TABLES:
        path = data_dir / f"{table}.csv"
        if path.exists() and not force:
            continue
        url = f"https://raw.githubusercontent.com/{WALS_REPO}/{ref}/cldf/{table}.csv"
        with urllib.request.urlopen(url) as resp:
            path.write_bytes(resp.read())


def _read(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class WalsData:
    """Raw WALS tables: languages, the allowed codes per feature, and each language's values."""

    def __init__(self, data_dir: Path = DATA_DIR):
        download(data_dir)
        self.languages = {r["ID"]: r for r in _read(data_dir / "languages.csv")}
        self.feature_names = {r["ID"]: r["Name"] for r in _read(data_dir / "parameters.csv")}

        self.codes = defaultdict(list)  # feature -> code ids in WALS order
        self.code_names = {}
        for r in sorted(_read(data_dir / "codes.csv"), key=lambda r: int(r["Number"])):
            self.codes[r["Parameter_ID"]].append(r["ID"])
            self.code_names[r["ID"]] = r["Name"]

        self.values = defaultdict(dict)  # language id -> {feature: code id}
        for r in _read(data_dir / "values.csv"):
            self.values[r["Language_ID"]][r["Parameter_ID"]] = r["Code_ID"]


class FeatureSpace:
    """WALS restricted to a set of features, encoded as integers for fast comparison.

    matrix[i, j] is the index of language i's code for features[j] in codes[features[j]],
    or -1 if WALS has no value. extra_codes adds codes WALS does not have (e.g. the
    unattested order VXO for 84A); no WALS language takes them.
    """

    def __init__(self, wals: WalsData, features: list, extra_codes: dict = None,
                 extra_code_names: dict = None):
        extra_codes = extra_codes or {}
        self.wals = wals
        self.features = [f for f in features if f in wals.codes]
        self.col = {f: j for j, f in enumerate(self.features)}
        self.codes = {
            f: wals.codes[f] + [c for c in extra_codes.get(f, []) if c not in wals.codes[f]]
            for f in self.features
        }
        self.code_index = {f: {c: k for k, c in enumerate(cs)} for f, cs in self.codes.items()}
        self.code_names = {**wals.code_names, **(extra_code_names or {})}

        self.lang_ids = [l for l, vals in wals.values.items()
                         if l in wals.languages and any(f in vals for f in self.features)]
        self.matrix = np.full((len(self.lang_ids), len(self.features)), -1, dtype=int)
        for i, lang in enumerate(self.lang_ids):
            for f, code in wals.values[lang].items():
                if f in self.col and code in self.code_index[f]:
                    self.matrix[i, self.col[f]] = self.code_index[f][code]

        genera = [wals.languages[l]["Genus"] or l for l in self.lang_ids]
        genus_ids = {g: k for k, g in enumerate(sorted(set(genera)))}
        self.genus = np.array([genus_ids[g] for g in genera])

    def language_name(self, i: int) -> str:
        lang = self.wals.languages[self.lang_ids[i]]
        return f"{lang['Name']} ({lang['Genus']}, {lang['Family']})"

    def code_name(self, code: str) -> str:
        return self.code_names.get(code, code)

    def encode(self, profile) -> np.ndarray:
        """Profile -> vector of code indices over self.features (-1 where missing)."""
        vec = np.full(len(self.features), -1, dtype=int)
        for f, answer in profile.answers.items():
            if f in self.col:
                vec[self.col[f]] = self.code_index[f][answer.code]
        return vec


if __name__ == "__main__":
    download(force=True)
    print(f"Downloaded WALS tables at {WALS_REF[:7]} into {DATA_DIR}")
