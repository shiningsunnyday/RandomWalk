#!/usr/bin/env bash
# Rebuild both revised documents from the submitted originals in one step, so a revision is always a function of
# inputs/submitted/*.docx + apply_main.py + apply_si.py + outputs/ and never of a hand-edited intermediate.
#
#   usage: bash results_v2/build_docx.sh [out_dir]      (default: a fresh timestamped dir under the scratchpad)
#
# Writes <out_dir>/out/{main.docx,si.docx} and validates both. Installing them over the working copies in
# ~/RandomWalk/ is deliberately NOT done here: do it yourself once the validator is clean.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
SK=$HOME/.claude/skills/synced/e74f2436-49a1-443d-a7d2-6fdff303f2d8_9707aade-dcf2-44c7-b6af-44341b9874ad/docx
PY=${DOCX_PY:-/tmp/claude-1026/-home-msun415/e4c142f8-9c54-42f0-bb59-09e87958531b/scratchpad/docx_venv/bin/python}
B=${1:-/tmp/claude-1026/-home-msun415/e4c142f8-9c54-42f0-bb59-09e87958531b/scratchpad/docx_build_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$B/main" "$B/si" "$B/out"
cd "$B"
unzip -q "$H/inputs/submitted/main_submitted.docx" -d "$B/main"
unzip -q "$H/inputs/submitted/si_submitted.docx" -d "$B/si"
find "$B/main" "$B/si" -type l -delete
$PY "$SK/scripts/merge_runs.py" "$B/main"
$PY "$SK/scripts/merge_runs.py" "$B/si"

$PY "$H/apply_main.py" "$B/main"
$PY "$H/apply_si.py" "$B/si"

# Figure 5 is now a PNG, so the EMF it replaced is unreferenced and the validator rejects it.
rm -f "$B/main/word/media/image5.emf"

(cd "$B/main" && zip -Xqr "$B/out/main.docx" .)
(cd "$B/si" && zip -Xqr "$B/out/si.docx" .)
$PY "$SK/scripts/office/validate.py" "$B/out/main.docx" --original "$H/inputs/submitted/main_submitted.docx"
$PY "$SK/scripts/office/validate.py" "$B/out/si.docx" --original "$H/inputs/submitted/si_submitted.docx"
echo "built: $B/out/main.docx  $B/out/si.docx"
