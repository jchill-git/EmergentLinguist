"""Feature tests: each one takes UD-annotated sentences and returns a WALS-style value.

Two ways to add a test:
  1. Drop a script into ud_llm_tests/ that follows the ud_llm contract
     (`python <name>.py input.conllu output.json`, writing
     {"distribution": ..., "final_answer": ...}). It is picked up automatically.
     fetch_ud_llm_tests.py downloads more of these from matejklemen/ud_llm.
  2. Subclass FeatureTest, implement run(), and decorate it with @register_test,
     in a module listed in TEST_MODULES (e.g. word_order_tests.py).
"""

import importlib
import json
import re
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from annotation import write_conllu

UD_LLM_DIR = Path(__file__).resolve().parent / "ud_llm_tests"
# Modules whose @register_test classes are added to the suite.
TEST_MODULES = ["word_order_tests"]

# WALS names for the bundled ud_llm tests, used for display only.
WALS_NAMES = {
    "81A": "Order of Subject, Object and Verb",
    "81B": "Two Dominant Orders of Subject, Object and Verb",
    "82A": "Order of Subject and Verb",
    "83A": "Order of Object and Verb",
    "84A": "Order of Object, Oblique, and Verb",
    "85A": "Order of Adposition and Noun Phrase",
    "86A": "Order of Genitive and Noun",
    "87A": "Order of Adjective and Noun",
    "88A": "Order of Demonstrative and Noun",
    "89A": "Order of Numeral and Noun",
    "90A": "Order of Relative Clause and Noun",
    "94A": "Order of Adverbial Subordinator and Clause",
    "144A": "Position of Negative Word With Respect to Subject, Object, and Verb",
    "144B": "Position of Negative Words Relative to Clause and Verb",
}


@dataclass
class TestResult:
    test_id: str
    name: str
    final_answer: str
    distribution: dict = field(default_factory=dict)
    error: str = None

    @property
    def n_observations(self) -> int:
        return sum(v for v in self.distribution.values() if isinstance(v, (int, float)))


class FeatureTest(ABC):
    test_id = "base"
    name = ""

    @abstractmethod
    def run(self, sentences: list) -> TestResult:
        """sentences is a list of conllu.TokenList."""


class UDLLMScriptTest(FeatureTest):
    """Runs an unmodified ud_llm extraction script on the sentences."""

    def __init__(self, script_path: Path):
        self.script_path = Path(script_path)
        self.test_id = self.script_path.stem
        self.name = WALS_NAMES.get(self.test_id, self.test_id)

    def run(self, sentences: list) -> TestResult:
        # The scripts write side files next to their input, so run them in a temp dir.
        with tempfile.TemporaryDirectory() as tmp:
            in_path, out_path = Path(tmp) / "input.conllu", Path(tmp) / "output.json"
            write_conllu(sentences, in_path)
            proc = subprocess.run(
                [sys.executable, str(self.script_path), str(in_path), str(out_path)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0 or not out_path.exists():
                err = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "no output"
                return TestResult(self.test_id, self.name, "ERROR", error=err)
            result = json.loads(out_path.read_text(encoding="utf-8"))
        return TestResult(self.test_id, self.name,
                          str(result.get("final_answer")), dict(result.get("distribution", {})))


_CUSTOM_TESTS = []


def register_test(cls):
    """Class decorator for FeatureTest subclasses written in Python."""
    if cls not in _CUSTOM_TESTS:
        _CUSTOM_TESTS.append(cls)
    return cls


def _sort_key(test_id: str):
    m = re.match(r"(\d+)([A-Za-z]*)", test_id)
    return (int(m.group(1)), m.group(2)) if m else (float("inf"), test_id)


def discover_tests(only: list = None) -> list:
    """All ud_llm scripts plus registered Python tests, sorted by WALS code.

    only optionally restricts to the given test ids (e.g. ["81A", "87A"]).
    """
    for module in TEST_MODULES:
        importlib.import_module(module)
    tests = [UDLLMScriptTest(p) for p in UD_LLM_DIR.glob("*.py")]
    tests += [cls() for cls in _CUSTOM_TESTS]
    if only:
        tests = [t for t in tests if t.test_id in only]
    return sorted(tests, key=lambda t: _sort_key(t.test_id))
