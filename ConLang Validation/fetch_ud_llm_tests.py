"""Download WALS extraction scripts from matejklemen/ud_llm into ud_llm_tests/.

The scripts are stored unmodified. Each one is run as
`python <code>.py input.conllu output.json` and writes
{"distribution": ..., "final_answer": ...} to output.json.

Usage:
    python fetch_ud_llm_tests.py                  # the default 13 word-order tests
    python fetch_ud_llm_tests.py 95A 96A 97A      # add more by WALS code
    python fetch_ud_llm_tests.py --ref main 81A   # pull from a different commit/branch
"""

import sys
import urllib.request
from argparse import ArgumentParser
from pathlib import Path

REPO = "matejklemen/ud_llm"
# Commit the bundled scripts were taken from.
DEFAULT_REF = "e569fe28358d1d2df84a04a2950b16df1aa8dc08"
SCRIPT_DIR = "WALS_python_extraction_scripts_gold"

# The 13 features the ud_llm paper evaluates (the ones with prompts in the repo).
DEFAULT_CODES = [
    "81A", "82A", "83A", "84A", "85A", "86A", "87A",
    "88A", "89A", "90A", "94A", "144A", "144B",
]

OUT_DIR = Path(__file__).resolve().parent / "ud_llm_tests"


def fetch(code: str, ref: str) -> Path:
    url = f"https://raw.githubusercontent.com/{REPO}/{ref}/{SCRIPT_DIR}/{code}.py"
    out_path = OUT_DIR / f"{code}.py"
    with urllib.request.urlopen(url) as resp:
        out_path.write_bytes(resp.read())
    return out_path


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("codes", nargs="*", default=DEFAULT_CODES,
                        help="WALS feature codes to download")
    parser.add_argument("--ref", default=DEFAULT_REF,
                        help="git commit or branch of ud_llm to download from")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    for code in args.codes:
        try:
            print(f"fetched {fetch(code, args.ref).name}")
        except Exception as e:
            print(f"failed  {code}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
