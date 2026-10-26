"""Run the WALS feature tests on plain-language sentences and their conlang translation.

Pipeline:
  1. Read plain-language sentences (default: ../sentences.txt).
  2. Parse them into UD annotation with spacy-stanza (cached in results/english.conllu).
  3. Translate the annotated sentences with the chosen translator.
  4. Run every feature test on both the English and the conlang sentences.
  5. Convert the test results into WALS answers and score the conlang for
     novelty (novelty.py) and plausibility (plausibility.py), alongside the
     reference cases in references.py.
  6. Print a report and save results/<translator>/results.json plus the
     translated sentences as results/<translator>/conlang.conllu.

Usage:
    python run_validation.py
    python run_validation.py --translator dummy --seed 3
    python run_validation.py --tests 81A 87A --no-wals
"""

import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataclasses import asdict
from pathlib import Path

from annotation import load_or_parse, write_conllu
from feature_tests import discover_tests
from references import headline, random_profiles, score_profile, summarize
from translators import TRANSLATORS, get_translator
from wals import FeatureSpace, WalsData
from wals_mapping import (DOMINANCE_RATIO, FEATURE_MAPS, extra_code_names, extra_codes,
                          profile_from_results)

HERE = Path(__file__).resolve().parent


def get_args():
    parser = ArgumentParser(description=__doc__.splitlines()[0],
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sentences", default=str(HERE.parent / "sentences.txt"),
                        help="plain-language sentences, one per line")
    parser.add_argument("--translator", default="dummy", choices=sorted(TRANSLATORS))
    parser.add_argument("--seed", type=int, default=0, help="seed passed to the translator")
    parser.add_argument("--tests", nargs="*", default=None,
                        help="only run these test ids (default: all discovered tests)")
    parser.add_argument("--out-dir", default=str(HERE / "results"))
    parser.add_argument("--examples", type=int, default=3,
                        help="number of example translations to print")

    wals = parser.add_argument_group("WALS scoring")
    wals.add_argument("--no-wals", action="store_true", help="skip novelty and plausibility scoring")
    wals.add_argument("--min-obs", type=int, default=5,
                      help="observations a test needs before its answer is used")
    wals.add_argument("--dominance-ratio", type=float, default=DOMINANCE_RATIO,
                      help="how many times more frequent an order must be to count as dominant")
    wals.add_argument("--min-overlap", type=int, default=3,
                      help="feature groups a WALS language must share with the profile to be compared")
    wals.add_argument("--include-same-genus", action="store_true",
                      help="let real languages match relatives when building the novelty percentile")
    wals.add_argument("--alpha", type=float, default=0.5,
                      help="count added to every answer when estimating probabilities")
    wals.add_argument("--rare-threshold", type=float, default=0.05,
                      help="pairs with P(answer | other answer) below this are listed as rare")
    wals.add_argument("--reference-seeds", type=int, default=5,
                      help="translator seeds (0..N-1) to score as a reference")
    wals.add_argument("--random-profiles", type=int, default=100,
                      help="random profiles to score as a reference")
    return parser.parse_args()


def run_tests(tests, sentences) -> dict:
    return {t.test_id: t.run(sentences) for t in tests}


def format_distribution(dist: dict) -> str:
    return ", ".join(f"{k}: {v}" for k, v in dist.items() if v) or "no observations"


def display_answer(result) -> str:
    # Some ud_llm scripts still return a category when nothing was observed (e.g. 90A).
    if result.error:
        return "ERROR"
    return result.final_answer if result.n_observations else "no data"


def print_report(tests, english_results, conlang_results, translator_name):
    print(f"\n{'Test':<6} {'English':<28} {translator_name:<28} {'n (en/conlang)':<14} Feature")
    print("-" * 110)
    for test in tests:
        en, cl = english_results[test.test_id], conlang_results[test.test_id]
        n = f"{en.n_observations}/{cl.n_observations}"
        print(f"{test.test_id:<6} {display_answer(en)[:27]:<28} {display_answer(cl)[:27]:<28} {n:<14} {test.name}")

    print("\nConlang distributions:")
    for test in tests:
        cl = conlang_results[test.test_id]
        detail = cl.error if cl.error else format_distribution(cl.distribution)
        print(f"  {test.test_id:<6} {detail}")


def print_profile(space, profile, args):
    print(f"\nWALS profile (min {args.min_obs} observations; dominant = "
          f"{args.dominance_ratio:g}x more frequent than the next answer):")
    for f, answer in profile.answers.items():
        print(f"  {f:<6} {space.code_name(answer.code):<34} n={answer.n}")
    for f, reason in profile.skipped.items():
        print(f"  {f:<6} {'(not used)':<34} {reason}")


def print_novelty(nov: dict):
    print("\nNovelty")
    if "error" in nov:
        print(f"  not computed: {nov['error']}")
        return
    c, s = nov["closest"], nov["soft_closest"]
    print(f"  Using {len(nov['features'])} features in {nov['n_groups']} groups")
    print(f"  A. Closest language: {c['language']}, distance {c['distance']:.2f} "
          f"over {c['shared_features']} shared features")
    for d in c["differences"]:
        print(f"       differs on {d['feature']}: conlang {d['profile']} vs. {d['language']}")
    genus_note = "relatives excluded" if nov["exclude_same_genus"] else "relatives allowed"
    print(f"  B. More distinctive than {nov['percentile']:.0f}% of {nov['n_reference']} real languages "
          f"(each compared with its closest language, {genus_note})")
    print(f"  C. Exact matches: {nov['exact_matches']} of {nov['n_candidates']} comparable languages agree "
          f"on every shared feature ({nov['exact_matches_all_features']} have data for all features)")
    print(f"  D. Closest using proportions: {s['language']}, distance {s['distance']:.2f}")


def print_plausibility(pla: dict):
    print("\nPlausibility (natural log probabilities; 0 is best)")
    if "error" in pla:
        print(f"  not computed: {pla['error']}")
        return
    print(f"  E. Mean log probability per feature group: {pla['feature_logprob_mean']:.2f} "
          f"(sum {pla['feature_logprob_sum']:.2f} over {pla['n_groups']} groups, {len(pla['features'])} features)")
    for x in pla["per_feature"]:
        print(f"       {x['feature']:<6} {x['answer']:<34} P={x['prob']:.3f}")
    if pla["pair_logprob_mean"] is None:
        print("  F. Not computed: needs features from at least two groups")
        return
    print(f"  F. Mean log probability per feature pair: {pla['pair_logprob_mean']:.2f} "
          f"over {pla['n_pairs']} ordered pairs")
    if pla["rare_pairs"]:
        print(f"     Rare combinations (P < {pla['rare_threshold']:g}):")
        for x in pla["rare_pairs"]:
            print(f"       {x['given']}={x['given_answer']} -> {x['feature']}={x['answer']}: "
                  f"P={x['prob']:.3f} ({x['n_languages_both']} of {x['n_languages_given']} languages)")
    else:
        print(f"     No rare combinations (P < {pla['rare_threshold']:g})")


def _fmt(value, spec):
    return "-" if value is None else format(value, spec)


def print_references(rows: list):
    print("\nReference cases")
    print(f"  {'':<30} {'closest':>8} {'pctile':>7} {'exact':>6} {'E mean':>7} {'F mean':>7} {'rare':>5}")
    for label, h in rows:
        print(f"  {label:<30} {_fmt(h['distance'], '8.2f')} {_fmt(h['percentile'], '7.0f')} "
              f"{_fmt(h['exact_matches'], '6.0f')} {_fmt(h['feature_logprob_mean'], '7.2f')} "
              f"{_fmt(h['pair_logprob_mean'], '7.2f')} {_fmt(h['n_rare_pairs'], '5.1f')}")


def summary_row(summary: dict) -> dict:
    return {k: (v["mean"] if v else None) for k, v in summary.items()}


def score_wals(args, tests, english, english_results, conlang_results) -> dict:
    space = FeatureSpace(WalsData(), list(FEATURE_MAPS), extra_codes(), extra_code_names())

    def to_profile(results):
        return profile_from_results(results, args.min_obs, args.dominance_ratio)

    profile = to_profile(conlang_results)
    print_profile(space, profile, args)
    scores = score_profile(space, profile, args)
    print_novelty(scores["novelty"])
    print_plausibility(scores["plausibility"])

    english_scores = score_profile(space, to_profile(english_results), args)

    seed_headlines = []
    for seed in range(args.reference_seeds):
        translator = get_translator(args.translator, seed=seed)
        seed_profile = to_profile(run_tests(tests, translator.translate_all(english)))
        seed_headlines.append(headline(score_profile(space, seed_profile, args)))
    seed_summary = summarize(seed_headlines)

    random_headlines = [headline(score_profile(space, p, args))
                        for p in random_profiles(profile.features, args.random_profiles)]
    random_summary = summarize(random_headlines)

    rows = [(f"{args.translator} (seed {args.seed})", headline(scores)),
            ("English", headline(english_scores))]
    if seed_headlines:
        rows.append((f"{args.translator} seeds 0-{args.reference_seeds - 1} (mean)", summary_row(seed_summary)))
    if random_headlines:
        rows.append((f"random x{args.random_profiles} (mean)", summary_row(random_summary)))
    print_references(rows)
    print("  closest = distance to closest language (0 = some language matches exactly); "
          "pctile = novelty percentile;\n  exact = exact-match languages; E/F mean = plausibility; "
          "rare = number of rare combinations.\n  Random profiles use the same features as the conlang.")

    return {
        "settings": {k: getattr(args, k) for k in [
            "min_obs", "dominance_ratio", "min_overlap", "include_same_genus",
            "alpha", "rare_threshold", "reference_seeds", "random_profiles"]},
        "profile": {"answers": {f: asdict(a) for f, a in profile.answers.items()},
                    "skipped": profile.skipped},
        **scores,
        "references": {
            "english": english_scores,
            "translator_seeds": {"headlines": seed_headlines, "summary": seed_summary},
            "random": {"summary": random_summary},
        },
    }


def main():
    args = get_args()
    out_dir = Path(args.out_dir)

    english = load_or_parse(args.sentences, out_dir / "english.conllu")
    print(f"Loaded {len(english)} annotated sentences from {args.sentences}")

    translator = get_translator(args.translator, seed=args.seed)
    conlang = translator.translate_all(english)

    translator_dir = out_dir / args.translator
    translator_dir.mkdir(parents=True, exist_ok=True)
    write_conllu(conlang, translator_dir / "conlang.conllu")

    print("\nExample translations:")
    for en, cl in list(zip(english, conlang))[:args.examples]:
        print(f"  {en.metadata['text']}\n  -> {cl.metadata['text']}")

    tests = discover_tests(args.tests)
    english_results = run_tests(tests, english)
    conlang_results = run_tests(tests, conlang)

    print_report(tests, english_results, conlang_results, args.translator)

    output = {
        "sentences": args.sentences,
        "translator": args.translator,
        "seed": args.seed,
        "tests": {
            t.test_id: {
                "name": t.name,
                "english": asdict(english_results[t.test_id]),
                "conlang": asdict(conlang_results[t.test_id]),
            } for t in tests
        },
    }
    if not args.no_wals:
        output["wals"] = score_wals(args, tests, english, english_results, conlang_results)

    with open(translator_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {translator_dir}")


if __name__ == "__main__":
    main()
