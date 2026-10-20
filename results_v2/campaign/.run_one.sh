#!/usr/bin/env bash
H=$1; ARM=$2; shift 2
PCK=$1; shift
PC="${PCK:0:1} ${PCK:1:1}"
s=$1; ew=$2; fw=$3; init=$4; dir=$5; ep=$6
tag="p${PCK}_ew${ew}_fw${fw}_${init}_dir${dir}_s${s}_e${ep}"
bash "$H/train.sh" "$PC" "$@" > "$H/logs/$ARM/$tag.log" 2>&1
echo "$tag rc=$?"
