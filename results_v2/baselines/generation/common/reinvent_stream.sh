#!/bin/bash
# Draw a sample stream from one of the two REINVENT 4 agents, for the unified protocol (common/unified.py).
# The agent file is the one the original 1000-sample run used, so the model is identical across the two tables;
# only num_smiles changes. usage: reinvent_stream.sh {tl|rl} <n>
set -euo pipefail
STAGE=$1; N=$2
cd "$(dirname "$0")/../runs/reinvent"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
sed -e "s/^num_smiles = .*/num_smiles = $N/" \
    -e "s/^output_file = .*/output_file = \"stream_${STAGE}.csv\"/" \
    -e "s/^json_out_config = .*/json_out_config = \"_stream_${STAGE}.json\"/" \
    "sample_${STAGE}.toml" > "stream_${STAGE}.toml"
nice -n 10 /home/msun415/miniconda3/envs/reinvent4/bin/reinvent -l "stream_${STAGE}.log" "stream_${STAGE}.toml" \
  > "stream_${STAGE}_stdout.log" 2>&1
echo "EXIT=$? rows=$(wc -l < "stream_${STAGE}.csv")"
