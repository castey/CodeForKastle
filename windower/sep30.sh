#!/bin/bash

# Run windower with different entity sizes, distributions, and depths

ENTITY_SIZES=(10 100 1000 10000)
DISTRIBUTIONS=("uniform" "normal")
WINDOW_DEPTHS=(4 8 16)

for d in "${WINDOW_DEPTHS[@]}"; do
  for n in "${ENTITY_SIZES[@]}"; do
    for dist in "${DISTRIBUTIONS[@]}"; do
      outfile="kg_${dist}_${n}_depth${d}.ttl"
      echo "Generating: $outfile"
      python windower.py \
        --n_entities "$n" \
        --depth "$d" \
        --D "$dist" \
        --low 0 \
        --high 1 \
        --sort_input False \
        --verbose False \
        --save "$outfile" \
        --precision 1e-4
    done
  done
done

echo "All runs complete."
