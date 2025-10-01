#!/bin/bash

# Run windower with different entity sizes and distributions

ENTITY_SIZES=(10 100 1000 10000)
DISTRIBUTIONS=("uniform" "normal")

for n in "${ENTITY_SIZES[@]}"; do
  for dist in "${DISTRIBUTIONS[@]}"; do
    outfile="kg_${dist}_${n}_depth4.ttl"
    echo "Generating: $outfile"
    python windower.py \
      --n_entities "$n" \
      --depth 4 \
      --D "$dist" \
      --low 0 \
      --high 10 \
      --sort_input False \
      --verbose True \
      --save "$outfile"
  done
done

echo "All runs complete."
