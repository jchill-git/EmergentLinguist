# Word Order (Group 2)

The word order functions live in `conlang_transformation_functions.py`, in the section marked `WORD ORDER (Janet)`. There is one function per highlighted WALS feature, and every function works on the `TokenRecord` list. `sentence_transform()` calls all five through `apply_word_order`.

| WALS | Function | Value | English → conlang (word order only) |
|---|---|---|---|
| 81A | `sov_81A` | SOV | The dog ate the bone. → The dog the bone ate. |
| 85A | `postpositions_85A` | Postpositions | The cat sat on the mat. → The cat sat the mat on. |
| 86A | `genitive_noun_86A` | Genitive-Noun | It is the last hour of the year. → It is the year the last hour. |
| 87A | `noun_adjective_87A` | Noun-Adjective | A big brown dog barked. → A dog big brown barked. |
| 90A | `relcl_noun_90A` | Relative clause-Noun | The book that I read was long. → I read The book was long. |

## Inputs and outputs

Every function has the same signature:

```
f(records: list[TokenRecord], **options) -> list[TokenRecord]
```

- **Input:** TokenRecords in their current order (by `current_index`). They can come straight from `build_token_records(nlp(sentence))` or from an earlier step. Fields read: `surface`, `lemma`, `pos`, `tag`, `dep`, `original_index`, `head_index`, `current_index`. The input list is not modified.
- **Output:** a new list of copied TokenRecords in conlang order, with `current_index` renumbered 0..n-1.
  - `surface`, `lemma`, `pos`, `tag` and `markers` are unchanged.
  - Words the conlang does not write are left out of the list, the same way `passive_to_active` removes "by". These are "of" in a genitive, who/which/that in a relative clause, and the commas around a non-restrictive relative clause.
  - When "of" is left out, its object is re-attached to the possessed noun with `dep = "poss"`. Every `head_index` still points to a word in the list.

The dependency tree is rebuilt from `head_index`; nothing is re-parsed. Each function only reorders the direct dependents of one kind of head (verbs for 81A, adpositions for 85A, nouns for 86A/87A/90A), and every dependent moves together with its whole phrase. Inside a noun phrase the order is fixed:

```
[genitive] [noun-modifying PP] [relative clause] [the/a, compounds] NOUN [adjectives] [other]
```

so the five functions give the same result in any order (checked on the 99 sample sentences). `WORD_ORDER_FEATURES` maps each WALS ID to its function, so `apply_word_order(records, ["81A", "87A"])` runs only those.

Records added by other steps (`original_index = -1`) are moved like any other word attached to their head. This covers the question particle and the placeholder subject of a truncated passive. The question particle (dep `discourse`) stays at the start of the sentence.

| Function | Reads | Options (default) |
|---|---|---|
| `sov_81A` | `pos` VERB/AUX heads; `dep` nsubj(pass), csubj, expl, dobj/obj, attr, acomp, oprd, ccomp, xcomp, dative/iobj, aux(pass), neg, prt, punct, discourse; `tag` WRB | `obliques_before_object=True` (XOV, 84A), `auxiliaries_after_verb=True` |
| `postpositions_85A` | `dep` prep, agent, dative (ADP) + pobj/pcomp; head `pos` | `adnominal_pp_before_noun=True` |
| `genitive_noun_86A` | `dep` poss, prep "of" + pobj, case; `surface`, `lemma`, `pos` | `drop_of=True`, `drop_possessive_marker=False` |
| `noun_adjective_87A` | `dep` amod, head `pos` | – |
| `relcl_noun_90A` | `dep` relcl, `tag` WDT/WP | `drop_relativizer=True` |

## Place in `sentence_transform`

```
build_token_records(nlp(sentence))
→ yes_no_to_particle → passive_to_active        # sentence rewrites
→ apply_word_order                              # word order (81A, 85A, 86A, 87A, 90A)
→ mark_causative_and_reorder                    # clause shuffle keeps each clause's order
→ mark_accusatives → mark_applicative           # markers
→ render_sentence
```

Word order runs after the passive rewrite, so the new active sentence gets SOV. It runs before the causative shuffle, which moves whole clauses and keeps the SOV order inside each clause.

## Checking the output

Run `sample_sentence_transformation.ipynb` top to bottom. Besides Joe's cells, it has two cells for word order:
- **word order only:** runs `apply_word_order` on one sentence, without the other steps
- **side by side:** prints each sentence in `sentences.txt` next to its transformed version, to check each rule

The notebook also writes `transformed_sentences.txt`.

## Known spaCy (en_core_web_sm) parse errors

These are parse errors, not rule bugs:
- *The man who saw the dog left.*: "the dog left" is read as a clause under "saw".
- *The sheep rest in the field.*: "rest" is read as a noun, so the output is *The field in the sheep rest.*

To see a sentence's labels: `[(t.text, t.dep_, t.head.text) for t in nlp(text)]`.
