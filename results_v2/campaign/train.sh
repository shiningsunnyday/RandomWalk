#!/usr/bin/env bash
# Campaign predictor run. Same protocol as results_v2/train_predictor.sh (validation-selected checkpoint,
# --train_norm, test split only evaluated) but with the campaign code copy and extra backward-compatible flags.
# usage: train.sh "<property_cols>" <split_seed> <ew 0|1> <fw 0|1> <init> <directed 0|1> <epochs> <arm> [extra flags...]
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$H/.." && pwd)
PC=$1; SEED=$2; EW=$3; FW=$4; INIT=$5; DIR=$6; EPOCHS=$7; ARM=$8; shift 8
export TORCH_SEED=${TORCH_SEED:-0} PYTHONHASHSEED=0 OMP_NUM_THREADS=1 PYTHONBREAKPOINT=0 PYTHONDONTWRITEBYTECODE=1
export dataset=$R/data/group-contrib
STUBS=${STUBS:-/tmp/claude-1026/-home-msun415-RandomWalk/e4c142f8-9c54-42f0-bb59-09e87958531b/scratchpad/stubs}
PY=${PY:-$HOME/miniconda3/envs/polymer_walk/bin/python}
$PY -c "import moses.metrics, eden.graph" 2>/dev/null || export PYTHONPATH=$STUBS
FPSUF=""
[ "${FP_BITS:-2048}" != 2048 ] || [ "${FP_RADIUS:-2}" != 2 ] && FPSUF="_fp${FP_BITS:-2048}r${FP_RADIUS:-2}"
TAG=p${PC// /}_ew${EW}_fw${FW}_${INIT}_dir${DIR}_s${SEED}_t${TORCH_SEED}${FPSUF}_e${EPOCHS}
G=$H/outputs/$ARM/$TAG; mkdir -p "$G"
for f in ckpt.pt config.json; do [ -e "$G/$f" ] || ln -s "$R/grammar/$f" "$G/$f" 2>/dev/null || true; done
EWF=""; [ "$EW" = 1 ] && EWF="--edge_weights"
FWF=""; [ "$FW" = 1 ] && FWF="--feat_concat_W"
DF=""; [ "$DIR" = 1 ] && DF="--ablate_bidir"
cd "$H/code"
exec $PY main.py \
 --motifs_folder $dataset/all_groups/ --extra_label_path $dataset/all_groups/all_extra.txt \
 --grammar_folder "$G" --cuda -1 --predefined_graph_file $dataset/red_graph.edgelist \
 --dags_file $dataset/dags.pkl --walks_file $dataset/polymer_walks_v2_preprocess.txt \
 --property_cols $PC --norm_metrics --train_norm --test_seed $SEED --train_size 1.0 --test_size 0.2 --val_size 0.2 \
 --task regression --num_epochs $EPOCHS --batch_size 1 --num_workers 0 --prefetch_factor 0 --lr 0.001 --grammar_lr 0.0 \
 --momentum 0.0 --num_accumulation_steps 1 --opt adam --gamma 0.1 --shuffle --share_params --mol_feat fp \
 --hidden_dim 16 --num_layers 5 $EWF $FWF $DF --gnn gin --act relu --dropout_rate 0.0 --num_transformer_heads 1 \
 --init $INIT --num_generate_samples 15 "$@"
