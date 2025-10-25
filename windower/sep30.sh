#!/bin/bash

# Run windower with different entity sizes, distributions, depths, and lt_modes

ENTITY_SIZES=(1000)
DISTRIBUTIONS=("uniform")
WINDOW_DEPTHS=(4 8 16)
LT_MODES=("rand5")

for d in "${WINDOW_DEPTHS[@]}"; do
  for n in "${ENTITY_SIZES[@]}"; do
    for dist in "${DISTRIBUTIONS[@]}"; do
      for mode in "${LT_MODES[@]}"; do
        echo "Generating: ${WINDOW_DEPTHS[@]}_${ENTITY_SIZES[@]}_${DISTRIBUTIONS[@]}_${LT_MODES[@]}"
        python windower.py \
          --n_entities "$n" \
          --depth "$d" \
          --D "$dist" \
          --low 0 \
          --high 1 \
          --verbose False \
          --outfile "$outfile" \
          --lt_mode "$mode" \
          --precision 4
      done
    done
  done
done

echo "All runs complete."
