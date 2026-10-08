#!/bin/bash
# Run SFT reasoning chain generation in waves using pi subagents.
# Each wave dispatches 5 parallel workers, each processing 1 batch of 20 pairs.
#
# Usage:
#   bash scripts/run_sft_generation.sh          # Run all remaining batches
#   bash scripts/run_sft_generation.sh 5 10     # Run batches 5-10 only

START=${1:-5}
END=${2:-254}
WAVE_SIZE=5
RESULTS_DIR="data/sft_reasoning/results"

echo "=== SFT Reasoning Chain Generation ==="
echo "Batches: $START to $END ($(( (END - START + 1) )) batches)"
echo "Wave size: $WAVE_SIZE parallel workers"
echo "Estimated waves: $(( (END - START + 1 + WAVE_SIZE - 1) / WAVE_SIZE ))"
echo ""

mkdir -p "$RESULTS_DIR"

for (( batch=$START; batch<=$END; batch+=$WAVE_SIZE )); do
    wave_end=$(( batch + WAVE_SIZE - 1 ))
    if [ $wave_end -gt $END ]; then
        wave_end=$END
    fi
    
    # Check which batches in this wave still need processing
    todo=()
    for (( b=$batch; b<=$wave_end; b++ )); do
        if [ ! -f "$RESULTS_DIR/batch_${b}.jsonl" ]; then
            todo+=($b)
        fi
    done
    
    if [ ${#todo[@]} -eq 0 ]; then
        continue
    fi
    
    echo "[Wave $(( (batch - START) / WAVE_SIZE + 1 ))] Batches: ${todo[*]}"
    
    # This is where you'd dispatch subagents — but since this is a shell script,
    # we just print the commands. The actual dispatch happens from the agent.
    for b in "${todo[@]}"; do
        echo "  → batch_$b (20 pairs)"
    done
    echo ""
done
