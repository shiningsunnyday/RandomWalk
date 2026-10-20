#!/usr/bin/env bash
# Train the random-walk grammar on the corrected dataset with the paper's recipe (diffusion.py: context_L,
# alpha 1e-3, Adam, 2000 epochs), seeded. Writes results_v2/outputs/grammar/logs/logs-<time>/ (ckpt.pt, config.json,
# L_loss.png) and symlinks it as results_v2/outputs/grammar/current.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
export TORCH_SEED=${TORCH_SEED:-0} PYTHONHASHSEED=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} PYTHONBREAKPOINT=0 PYTHONDONTWRITEBYTECODE=1
export dataset=$H/data/group-contrib
STUBS=${STUBS:-/tmp/claude-1026/-home-msun415-RandomWalk/e4c142f8-9c54-42f0-bb59-09e87958531b/scratchpad/stubs}
PY=${PY:-$HOME/miniconda3/envs/polymer_walk/bin/python}
$PY -c "import moses.metrics, eden.graph" 2>/dev/null || export PYTHONPATH=$STUBS
mkdir -p "$H/outputs/grammar" && cd "$H/outputs/grammar"
$PY "$H/code/diffusion.py" --data_file $dataset/polymer_walks_v2_preprocess.txt \
  --motifs_folder $dataset/all_groups/ --extra_label_path $dataset/all_groups/all_extra.txt \
  --dags_file $dataset/dags.pkl --predefined_graph_file $dataset/red_graph.edgelist \
  --num_epochs ${EPOCHS:-2000} --alpha 1e-3 --context_L --num_generate_samples 1
ln -sfn "$(ls -td logs/logs-* | head -1)" current
echo "grammar: $(readlink current)"
