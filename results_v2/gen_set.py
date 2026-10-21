"""
Which generated set the paper reports, in one place.

`gen_unique.py` is the same best-first enumeration as `gen.py bestfirst`, with one change: a closed walk whose
canonical cycle (over rotation and reflection) has already been accepted is rejected, so the budget is never spent
twice on the same repeat unit read from a different starting motif. It raises Unique from 62% to 75% and reaches as
many distinct molecules in 1000 walks as the plain enumeration reaches in 1400 (BRIEFING_2 §3a).

Switching STEM back to 'bestfirst_n1000' restores the originally submitted set everywhere: Table 1, Tables S8/S9,
Figs S2-S4 and the Table S14 row all read these paths. Nothing under outputs/ is overwritten either way; the two
sets live side by side.
"""
STEM = 'unique_n1000'                       # 'bestfirst_n1000' for the originally submitted enumeration
CLEAN = f'gen/{STEM}_clean.txt'             # <clean smiles> <sanitized 0|1> <walk>, one generated structure per line
WALKS = f'screening_walks_{"unique" if STEM.startswith("unique") else "bestfirst"}.txt'
ENS = f's9_ensemble{"_unique" if STEM.startswith("unique") else ""}.txt'
RETRO = f'retro/{"unique" if STEM.startswith("unique") else "bestfirst"}_clean_retro.txt'
