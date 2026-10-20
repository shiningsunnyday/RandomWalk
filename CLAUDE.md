# Working notes for this repo

Revision work for the gas-separation membrane paper lives in `results_v2/`. Everything below was learned the hard
way; please keep it current.

## Git

Commit regularly, not just when asked. Commit messages are short and lowercase, two to five words, matching the
existing log (`fix correspondence and dedup`, `minor bug fixes`, `metrics`). No body unless something genuinely
needs explaining.

The repo has a lot of pre-existing untracked clutter at the root (`datasets/`, `feat_cache/`, `__pycache__/`,
PDFs, scratch `.txt` result dumps). Commit the files the current work touches; do not sweep all of it in.

## Do not touch

- **The 97 canonical motif groups are never deleted or renumbered.** Modify in place or append only. There are 99
  group files now; 1–97 are the canonical set.
- `results_v2/outputs/`, `results_v2/grammar/`, `results_v2/data/`, `results_v2/code/` are frozen. The campaign
  keeps its own patched copy at `results_v2/campaign/code/`; patch there instead. To change generation behaviour
  without forking `code/diffusion.py`, wrap a module-level function, as `results_v2/gen_unique.py` does.
- **Experimental results stay as they are.** Table S1 is preliminary and must not be overwritten. Table 3 and
  Tables S3/S5/S7 are experimental and stay untouched.
- Do not modify existing conda envs in place.
- We are submitting to a new venue, so no reviewer-response document is needed.

## Environments

`python` on PATH is ADFRsuite python2. Always name an interpreter explicitly.

| what | interpreter |
|---|---|
| grammar, predictor, generation, analysis | `~/miniconda3/envs/polymer_walk/bin/python` |
| chemprop, REINVENT | `~/miniconda3/envs/reinvent4/bin/python` |
| Retro\* planner | `~/miniconda3/envs/retro_star_env/bin/python` |
| transformers (polyBERT, ChemBERTa) | `~/miniconda3/envs/llamole/bin/python` |
| docx editing (lxml) | the py3.10 venv under the session scratchpad |

Grammar code needs stub packages on `PYTHONPATH` (a `moses`/`eden` shim). **Never run it with `python -I`** —
isolated mode ignores `PYTHONPATH` and the `moses` import fails. `pandoc` is **not** on PATH; extract docx text with
lxml instead.

## Document build

`results_v2/apply_main.py` and `apply_si.py` edit the submitted `.docx` files programmatically; every number they
write is generated from `outputs/numbers.json` or a `baselines/outputs/*.json`, never typed. So fixing a number
means fixing the script that computes it, then rebuilding.

```bash
cd results_v2 && bash build_docx.sh      # must print "All validations PASSED!" for both documents
```

Install to exactly these two paths:

- `AI-driven Discovery and Experimental Validation of Polymer Structures for Gas Separation Membranes - Submission Candidate.docx`
- `CSAIL_Paper_SI_V3.docx`

Main-text figures are EMFs with vectorised text, so figure labels cannot be edited without the source artwork.
A visible sentence may not exist as one XML string even after `merge_runs.py`, because a citation like `(42)` is its
own run.

## Evaluation protocol

Honest protocol: 3 split seeds (1,2,3), 80/20 train/test, 20% of train held out as validation, **validation alone**
selects epoch/config/checkpoint, `--train_norm`, `heldout_*` metrics. The campaign and the K=10 baseline re-runs use
seeds 1–10, which makes them directly comparable to each other.

**Rank by MAE, not R².** Measured, not asserted: at ten seeds 13 of 13 methods lose R² (median −0.114) while MAE
moves by at most 0.034 and the ranking holds at Kendall τ = 0.85. R²'s denominator is the test split's own variance,
so narrow splits produce the collapses.

## Provenance

`results_v2/PROVENANCE.md` records where every reported number comes from and the regeneration command for each.
Keep it current with any new script. `results_v2/BRIEFING_2.md` is the running brief to the author, including open
decisions and flags against the published paper.
