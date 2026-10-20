#!/usr/bin/env bash
# usage: run_arm.sh <jobs file> <arm name> <parallel>
# jobs file lines: <pc:01|21|43> <seed> <ew> <fw> <init> <dir> <epochs> <arm> [TSEED=n] [FPB=n] [FPR=n] [main.py flags...]
set -u
H=$(cd "$(dirname "$0")" && pwd)
J=$1; ARM=$2; P=${3:-60}
mkdir -p "$H/logs/$ARM"
IN="$H/.run_one_$ARM.sh"
cat > "$IN" <<'INNER'
#!/usr/bin/env bash
H=$1; ARM=$2; shift 2
PCK=$1; shift
PC="${PCK:0:1} ${PCK:1:1}"
s=$1; ew=$2; fw=$3; init=$4; dir=$5; ep=$6
args=()
for a in "$@"; do case "$a" in
  TSEED=*) export TORCH_SEED="${a#TSEED=}";;
  FPB=*)   export FP_BITS="${a#FPB=}";;
  FPR=*)   export FP_RADIUS="${a#FPR=}";;
  *) args+=("$a");;
esac; done
fpsuf=""
[ "${FP_BITS:-2048}" != 2048 ] || [ "${FP_RADIUS:-2}" != 2 ] && fpsuf="_fp${FP_BITS:-2048}r${FP_RADIUS:-2}"
tag="p${PCK}_ew${ew}_fw${fw}_${init}_dir${dir}_s${s}_t${TORCH_SEED:-0}${fpsuf}_e${ep}"
bash "$H/train.sh" "$PC" "${args[@]}" > "$H/logs/$ARM/$tag.log" 2>&1
echo "$tag rc=$?"
INNER
chmod +x "$IN"
xargs -P "$P" -L 1 "$IN" "$H" "$ARM" < "$J"
echo "ARM $ARM DONE $(date)"
