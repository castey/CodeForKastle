#!/bin/bash

# Run windower with different entity sizes, distributions, depths, and lt_modes

ENTITY_SIZES=(10 100 1000 10000)
DISTRIBUTIONS=("uniform" "normal")
WINDOW_DEPTHS=(4 8 16)
LT_MODES=("sequential" "pairwise")

for d in "${WINDOW_DEPTHS[@]}"; do
  for n in "${ENTITY_SIZES[@]}"; do
    for dist in "${DISTRIBUTIONS[@]}"; do
      for mode in "${LT_MODES[@]}"; do
        outfile="kg_${dist}_${n}_depth${d}_${mode}.nt"
        echo "Generating: $outfile"
        python windower.py \
          --n_entities "$n" \
          --depth "$d" \
          --D "$dist" \
          --low 0 \
          --high 1 \
          --verbose False \
          --outfile "$outfile" \
          --lt_mode "$mode" \
          --precision 1e-4
      done
    done
  done
done

echo "All runs complete."
