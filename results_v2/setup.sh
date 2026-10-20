#!/usr/bin/env bash
# Freeze the code + corrected dataset that the revised paper (v2) is computed from, into results_v2/:
#   code/      snapshot of the repo code (HEAD + uncommitted working-tree fixes; CODE_VERSION records both)
#   data/group-contrib/  corrected dataset: 99 groups (97 canonical, 11 edited in place, P42 + S33 appended),
#                        motif graph with all non-redundant orientations, 114 corrected walks, dags
# usage: bash results_v2/setup.sh   (refuses to overwrite an existing snapshot; delete code/ data/ to refresh)
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
H=$R/results_v2
cd "$H"
[ -d code ] && { echo "snapshot exists; remove results_v2/code and results_v2/data to refresh"; exit 1; }

mkdir -p code
( cd "$R" && cp -r main.py diffusion.py walk_grammar.py construct_graph.py utils models "$H/code/" )
find code -name __pycache__ -type d -prune -exec rm -rf {} +
mkdir -p code/data && touch code/data/__init__.py
{ echo "HEAD $(git -C "$R" rev-parse HEAD)"; echo "uncommitted changes (git diff HEAD):"; git -C "$R" diff HEAD -- main.py diffusion.py walk_grammar.py construct_graph.py utils models; } > CODE_VERSION

D=data/group-contrib
mkdir -p $D
cp -r "$R"/data/datasets/group-contrib/all_groups $D/
rm -rf $D/all_groups/with_label
cp "$R"/data/datasets/group-contrib/red_graph.edgelist "$R"/data/datasets/group-contrib/dags.pkl $D/
cp "$R"/data/polymer_walks_v2_preprocess.txt $D/polymer_walks_v2_preprocess.txt          # 114 corrected walks + properties
cp "$R"/data/polymer_walks_v2_smiles_regen.txt $D/polymer_walks_v2_smiles.txt            # corrected SMILES (novelty reference, Table S6)

( find data -type f -print0 | sort -z | xargs -0 md5sum; find code -type f -print0 | sort -z | xargs -0 md5sum ) > MANIFEST.md5
echo "results_v2 ready: $(wc -l < MANIFEST.md5) files frozen"
