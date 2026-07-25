#!/usr/bin/env bash
# Bundle logs/notes for scp home. Set DEMO and ISAACLAB if needed.

set -euo pipefail

DEMO="${DEMO:-$HOME/workspace/NVIDIADEMO}"
ISAACLAB="${ISAACLAB:-$HOME/workspace/IsaacLab}"
OUT="${OUT:-$HOME/workspace/nvidiademo_results_bundle}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE="$OUT/$STAMP"

mkdir -p "$BUNDLE/logs_sample" "$BUNDLE/notes"

echo "[info] DEMO=$DEMO"
echo "[info] ISAACLAB=$ISAACLAB"
echo "[info] BUNDLE=$BUNDLE"

# Machine info
if [[ -f "$DEMO/scripts/cloud_bootstrap.sh" ]]; then
  bash "$DEMO/scripts/cloud_bootstrap.sh" > "$BUNDLE/machine_info.txt" || true
else
  nvidia-smi > "$BUNDLE/machine_info.txt" 2>&1 || true
fi

# Notes from demo repo if filled
if [[ -f "$DEMO/results/isaac_run_notes.md" ]]; then
  cp "$DEMO/results/isaac_run_notes.md" "$BUNDLE/notes/"
fi
if [[ -f "$DEMO/results/isaac_run_notes.TEMPLATE.md" ]]; then
  cp "$DEMO/results/isaac_run_notes.TEMPLATE.md" "$BUNDLE/notes/"
fi

# Copy any curves already placed in DEMO/results
mkdir -p "$BUNDLE/curves"
if ls "$DEMO/results"/isaac_reach_train_curve.* >/dev/null 2>&1; then
  cp -v "$DEMO/results"/isaac_reach_train_curve.* "$BUNDLE/curves/" || true
fi
if ls "$DEMO/results"/isaac_reach_play.* >/dev/null 2>&1; then
  cp -v "$DEMO/results"/isaac_reach_play.* "$BUNDLE/curves/" || true
fi

# Sample Isaac logs (not full multi-GB dump)
if [[ -d "$ISAACLAB/logs" ]]; then
  echo "[info] sampling logs under $ISAACLAB/logs"
  # copy last few run dirs' small files
  mapfile -t RUNS < <(find "$ISAACLAB/logs" -type d -name "Isaac-Reach*" 2>/dev/null | tail -5)
  # fallback: newest dirs under rsl_rl
  if [[ ${#RUNS[@]} -eq 0 ]]; then
    mapfile -t RUNS < <(find "$ISAACLAB/logs" -mindepth 2 -maxdepth 3 -type d 2>/dev/null | tail -5)
  fi
  i=0
  for r in "${RUNS[@]:-}"; do
    [[ -z "$r" ]] && continue
    i=$((i + 1))
    dest="$BUNDLE/logs_sample/run_${i}"
    mkdir -p "$dest"
    # copy modest files only
    find "$r" -maxdepth 2 -type f \( \
      -name "*.txt" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" \
      -o -name "*.csv" -o -name "events.*" -o -name "*.md" \
    \) -size -20M -exec cp -t "$dest" {} + 2>/dev/null || true
    echo "$r" > "$dest/source_path.txt"
  done
else
  echo "[warn] no $ISAACLAB/logs — train may not have run yet"
fi

# Manifest
{
  echo "stamp=$STAMP"
  echo "demo=$DEMO"
  echo "isaaclab=$ISAACLAB"
  echo "host=$(hostname)"
  date -Is 2>/dev/null || date
  echo "--- tree ---"
  find "$BUNDLE" -type f | sort
} > "$BUNDLE/MANIFEST.txt"

# Tarball
mkdir -p "$OUT"
TAR="$OUT/nvidiademo_results_${STAMP}.tar.gz"
tar czf "$TAR" -C "$OUT" "$STAMP"
echo "[ok] bundle dir: $BUNDLE"
echo "[ok] tarball:    $TAR"
echo
echo "Download example (run on your laptop):"
echo "  scp user@CLOUD:$TAR /d/NVIDIADEMO/results_bundle/"
echo "Then extract and copy notes/curves into D:\\NVIDIADEMO\\results\\"
